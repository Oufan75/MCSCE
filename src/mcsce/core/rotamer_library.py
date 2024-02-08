"""
This file defines the function that reads the Dunbrack protein side chain rotamer library

Shapovalov, Maxim V., and Roland L. Dunbrack Jr. "A smoothed backbone-dependent rotamer library for proteins derived from adaptive kernel density estimates and regressions." Structure 19.6 (2011): 844-858.

Coded by Jie Li
Date created: Jul 28, 2021
Modified by Oufan Zhang to add ptm rotamers
"""

from pathlib import Path
import numpy as np
from mcsce.core.definitions import ptm_aa, ptm_h

_filepath = Path(__file__).resolve().parent  # folder
_library_path = _filepath.joinpath('data', 'SimpleOpt1-5', 'ALL.bbdep.rotamers.lib')
_ptm_library_path = {
        #"xd120": _filepath.joinpath('data', 'xd120_ptm.lib'),
        "bid120": _filepath.joinpath('data', 'bid120_ptm.lib'),
        "sidepro": _filepath.joinpath('data', 'sidepro_ptm.lib'),
        "bd30": _filepath.joinpath('data', 'bd30_ptm.lib'),
        }


def wrap_angles(rot_angle):
    more_mask = rot_angle > 180.
    rot_angle[more_mask] -= 360.
    less_mask = rot_angle < -180.
    rot_angle[less_mask] += 360.
    return rot_angle

def get_closest_angle(input_angle, degsep=10):
    """
    Find closest angle in [-180, +170] with n degree separations
    """
    if input_angle > 180 - degsep/2.:
        # the closest would be 180, namely -180
        return -180
    return round(input_angle/degsep)*degsep 

def get_floor_angle(input_angle, degsep=120):
    """
    Round to angle in [0, 360) at the floor of n degree separations
    """
    if input_angle == 360.:
        return 0
    return np.floor(input_angle/degsep)*degsep

def augment_std(data, probability_threshold=0.001):
    """
    augment_with_std: when set to True, the chi_(1,2) +- sigma are also taken as individual rotamers, and
    the probabilities for chi_i, chi_i + sigma, chi_i - sigma becomes 1/9 of the original probability of chi_i
    """
    assert len(data) % 2 == 1
    nchis = int((len(data) - 1)/2)
    prob = data[0]
    vals = data[1: nchis+1]
    sigs = data[-nchis:]
    if nchis == 1:
        new_prob = prob / 3
        if new_prob > probability_threshold and sigs[0] >= 1:
            return [[new_prob, vals[0] - sigs[0]], 
                    [new_prob, vals[0]], 
                    [new_prob, vals[0] + sigs[0]]]
        else:
            return [prob] + vals
    else:
        new_chi1 = [vals[0] - sigs[0], vals[0], vals[0] + sigs[0]] if sigs[0] >= 1 else [vals[0]]
        new_chi2 = [vals[1] - sigs[1], vals[1], vals[1] + sigs[1]] if sigs[1] >= 1 else [vals[1]]
        new_prob = prob / (len(new_chi1) * len(new_chi2))
        if new_prob > probability_threshold and len(new_chi1) > 1 and len(new_chi2) > 1:
            new_chi = [[new_prob, chi1, chi2] for chi1 in new_chi1 for chi2 in new_chi2]
            for row in new_chi:
                row.extend(vals[2:])
            return new_chi
        else:
            return [prob] + vals

def sample_torsion(data):
    nchis = int((len(data) - 1)/2)
    prob = data[0]
    vals = data[1: 1+nchis]
    sigs = data[-nchis:]
    return [prob] + list(np.random.normal(vals, sigs))


class DunbrakRotamerLibrary:
    """
    data structure: {(restype, psi, phi): [np.array<N, c>(N: number of rotamers, c: number of chi values, the rotamer values in degree), np.array<N>(probability)]}
    """
    def __init__(self, probability_threshold=0.001, augment_with_std=True) -> None:
        """
        probability_threshold: the minimum probability of a rotamer to be considered
        augment_with_std: when set to True, the chi_(1,2) +- sigma are also taken as individual rotamers, and 
        the probabilities for chi_i, chi_i + sigma, chi_i - sigma becomes 1/9 of the original probability of chi_i,
        following the implementation in Bhowmick, Asmit, and Teresa Head-Gordon. "A Monte Carlo method for generating side chain structural ensembles." Structure 23.1 (2015): 44-55.
        """
        self._data = {}
        self.prob_threshold = probability_threshold
        with open(_library_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    # comment line to be ignored
                    continue
                restype, phi, psi, _, _, _, _, _, prob, chi1, chi2, chi3, chi4, chi1_sigma, chi2_sigma, chi3_sigma, chi4_sigma = \
                    line.split()
                prob = float(prob)
                if prob > probability_threshold:
                    phi = int(phi)
                    psi = int(psi)
                    if phi == 180 or psi == 180:
                        # the -180 angle data should be the same
                        assert (restype, min(phi, -180), min(psi, -180)) in self._data
                        continue

                if restype in ["CYS", "SER", "THR", "VAL"]: 
                    # residues with only chi1
                    pdata = [prob, float(chi1), float(chi1_sigma)]
                elif restype in ["ASN", "ASP", "HIS", "ILE", "LEU", "PHE", "PRO", "TRP", "TYR"]:
                    # residues with chi1 and chi2
                    pdata = [prob, float(chi1), float(chi2), float(chi1_sigma), float(chi2_sigma)]
                elif restype in ["GLN", "GLU", "MET"]:
                    pdata = [prob, float(chi1), float(chi2), float(chi3), float(chi1_sigma), float(chi2_sigma), float(chi3_sigma)]
                else:
                    pdata = [prob, float(chi1), float(chi2), float(chi3), float(chi4), float(chi1_sigma), 
                         float(chi2_sigma), float(chi3_sigma), float(chi4_sigma)]

                if augment_with_std:
                    pdata = augment_std(pdata, probability_threshold)

                label = (restype, phi, psi)
                if isinstance(pdata[0], float):
                    pdata = [pdata]
                if label in self._data:
                    self._data[label].extend(pdata)
                else:
                    self._data[label] = pdata
        for item in self._data:
            v = np.array(self._data[item])
            self._data[item] = [wrap_angles(v[:, 1:]), v[:, 0]]


    def retrieve_torsion_and_prob(self, residue_type, phi, psi, ptmlib):
        if np.isnan(phi):
            # This is the first residue, which do not have phi, so use phi=-180 as default
            phi = -180
        if np.isnan(psi):
            # this is the last residue, which do not have psi, so use psi=-180 as default
            psi = -180

        if residue_type in ptm_aa:
            chis, probs = self._data[(ptm_aa[residue_type], get_closest_angle(phi), get_closest_angle(psi))]
            # phosphate protonation states
            if residue_type not in ptmlib._info:
                print(f"ptm residue rotamers of {residue_type} not provided, assumes rotamers of unmodified residue")
                return [chis, probs]
            # phosphate protonation states
            if residue_type in ['S1P', 'T1P', 'Y1P', 'H1D', 'H1E', 'H2E']:
                residue_type = ptm_h[residue_type]

            # search ptm library
            ptm_info = ptmlib.get_dependence(residue_type)
            nchis = ptm_info[1]
 
            if ptm_info[-1] == -1:
                # no dependence
                ptm_data = ptmlib.retrieve_torsion_and_prob(residue_type, -360.)
                ptm_probs = np.array(ptm_data)[:, 0]
                torsions = wrap_angles(np.array(ptm_data)[:, 1:])
                assert torsions.shape[1] == ptm_info[0]
                return [torsions, ptm_probs]
            elif ptm_info[-1] == 0:
                # backbone dependent
                ptm_data = ptmlib.retrieve_torsion_and_prob(residue_type, (phi, psi))
                ptm_probs = np.array(ptm_data)[:, 0]
                torsions = wrap_angles(np.array(ptm_data)[:, 1:])
                assert torsions.shape[1] == ptm_info[0]
                return [torsions, ptm_probs]
            
            dchis = chis[:, ptm_info[-1] - 1]
            new_chis = []
            new_probs = []
            for i in range(chis.shape[0]):
                ptm_data = ptmlib.retrieve_torsion_and_prob(residue_type, dchis[i])
                ptm_probs = np.array(ptm_data)[:, 0]
                torsions = wrap_angles(np.array(ptm_data)[:, 1:])
                #assert np.sum(ptm_probs) <= 1
                for j in range(len(ptm_probs)):
                    p = probs[i]*ptm_probs[j] 
                    if p > self.prob_threshold:
                        new_chis.append(np.concatenate((chis[i, :(ptm_info[0]-nchis)], torsions[j])))
                        new_probs.append(p)
            assert len(new_chis[0]) == ptm_info[0]
            return [np.array(new_chis), np.array(new_probs)]

        if residue_type in ["HID", "HIE", "HIP"]:
            residue_type = "HIS"

        return self._data[(residue_type, get_closest_angle(phi), get_closest_angle(psi))]


class ptmRotamerLib():
    """
    ptmlib_type: str choice in ["bid120", "sidepro", "bd30"]
    data structure: {(restype, depended chi): [np.array<N, c>(N: number of rotamers, c: (probability, rotamer values, sigma values in degree)]}
    info structure: {(restype: [total chis, number of additional chis, dependence])}
    """
    def __init__(self, ptmlib_type="bid120", probability_threshold=0.001, augment_with_std=False):
        self._data = {}
        self._info = {}
        self.ptmlib_type = ptmlib_type
        self.ptmlib_type = ptmlib_type

        # also read backbone-independent library to fill in missing data in backbone-dependent library
        libs = [ptmlib_type]
        if ptmlib_type.startswith("bd"):
            libs.append("bid120")
        for i, ptmlib_type in enumerate(libs):
            with open(_ptm_library_path[ptmlib_type]) as f:
                for line in f:
                    line = line.strip()
                    # read chi information
                    if len(line) == 5 and line.startswith("# "):
                        restype = line.split()[-1]
                        if i == 0:
                            self._info[restype] = []
                            continue
                    if i == 0:
                        if line.startswith("# Number of chi"):
                            self._info[restype].append(int(line.split()[-1]))
                            continue
                        if line.startswith("# Number of new chi"):
                            self._info[restype].append(int(line.split()[-1]))
                            continue
                        if line.startswith("# Dependence Backbone"):
                            self._info[restype].append(0)
                            continue
                        if line.startswith("# No Dependence"):
                            self._info[restype].append(-1)
                            continue
                        if line.startswith("# Dependence"):
                            self._info[restype].append(int(line.split()[-1]))
                            continue
                        if line.startswith("#") or len(line) == 0:
                            # other commented line
                            continue
                    # read in rotamers
                    if line.startswith(restype):
                        items = line.split()
                        if self._info[restype][-1] < 0 or i > 0:
                            label = (restype, -360.)
                            pdata = [float(n) for n in items[1:]]
                        elif self._info[restype][-1] > 0:
                            # with chi dependence
                            label = (restype, float(items[1]))
                            pdata = [float(n) for n in items[3:]]
                        else:
                            # backbone dependence
                            label = (restype, float(items[1]), float(items[2]))
                            pdata = [float(n) for n in items[3:]]
                        if augment_with_std:
                            pdata = augment_std(pdata, probability_threshold)
                        else:
                            pdata = sample_torsion(pdata)

                        if isinstance(pdata[0], float):
                            pdata = [pdata]
                        if label in self._data:
                            self._data[label].extend(pdata)
                        else:
                            self._data[label] = pdata


    def get_dependence(self, residue_type):
        return self._info[residue_type]

    def retrieve_torsion_and_prob(self, residue_type, dchi=-360):
        if self._info[residue_type][-1] < 0:
            return self._data[(residue_type, -360)]
        elif self._info[residue_type][-1] == 0:
            phi = get_closest_angle(dchi[0], degsep=30)
            psi = get_closest_angle(dchi[1], degsep=30)
            if phi == 180:
                phi = -180
            if psi == 180:
                psi = -180
            if (residue_type, phi, psi) not in self._data:
                return self._data[(residue_type, -360)]
            return self._data[(residue_type, phi, psi)]
        else:
            # convert to [0, 360) scale
            if dchi < 0: dchi += 360.
            rchi = get_floor_angle(dchi, 120)
            return self._data[(residue_type, rchi)]


if __name__ == "__main__":
    library = DunbrakRotamerLibrary()
    ptmlib = ptmRotamerLib("bid120")
    for resn in ["SEP", "M3L", "TPO", "ALY"]:
        print(library.retrieve_torsion_and_prob(resn, -73, 154.8, ptmlib))   
