"""Exception classes defined for the MCSCE package.
Borrowed from IDP Conformer Generator package (https://github.com/julie-forman-kay-lab/IDPConformerGenerator) developed by Joao M. C. Teixeira"""
from mcsce import contactus as CONTACTUS
from mcsce import log
from mcsce.core import count_string_formatters


class IDPConfGenException(Exception):
    r"""
    IDPConfGen base exception.

    Parameters
    ----------
    *args
        If the first element of args contains ``'{}'`` it will be
        used as :attr:`errmsg` base string.
        Else, ``args`` are used to feed the sting method ``.format()``
        for the default exception :attr:`errmsg`.

    errmsg : optional
        If given, overrides any previous parameter and the ``str``
        value of ``errmsg`` is used as the Exception message.
        Defaults to ``None``.

    Examples
    --------
    Uses the default errormsg.
    >>> err = IDPCalculator(var1, var2)

    >>> err = IDPConfGenException('An error happened: {}, {}', var1, var2)

    >>> err = IDPConfGenException('An error happened')

    >>> err = IDPConfGenException(errmsg='Custom error msg')

    >>> err = IDPConfGenException(errmsg='')

    """

    errmsg = 'An unknnown error as occurred. ' + CONTACTUS.contact_message

    def __init__(self, *args, errmsg=None):

        # IDPConfGenException(errmsg='Custom error msg')
        if errmsg is not None:
            assert isinstance(errmsg, str), f'wrong errmsg type: {type(errmsg)}'
            self.errmsg = errmsg
            self.args = []

        elif len(args) == count_string_formatters(self.errmsg):
            self.args = args

        else:
            assert count_string_formatters(args[0]) == len(args[1:]), \
                'args passed to Exception are not compatible to form a message'
            self.errmsg = args[0]
            self.args = args[1:]

        log.debug(f'Exception errors: {self.errmsg}')
        log.debug(f'Exception args: {self.args}')

        # ensure
        assert isinstance(self.args, (tuple, list)), \
            f'wrong args {type(self.args)}'
        assert count_string_formatters(self.errmsg) == len(self.args), (
            'Bad Exception message:\n'
            f'errmsg: {self.errmsg}\n'
            f'args: {self.args}'
            )

    def __str__(self):
        """Make me a string :-)."""
        return self.errmsg.format(*self.args)

    def __repr__(self):
        return f'{self.__class__.__name__}: {self}'

    def report(self):
        """
        Report error in the form of a string.

        Identifies Error type and error message.

        Returns
        -------
        str
            The formatted string report.
        """
        return f'{self.__class__.__name__} * {self}'


class PDBIDFactoryError(IDPConfGenException):
    """General PDBIDFactory Exception."""

    pass


class PDBFormatError(IDPConfGenException):
    """Exception for PDB format related issues."""


class CIFFileError(IDPConfGenException):
    """CIF file has loop_ but yet is invalid."""

    pass


class DownloadFailedError(IDPConfGenException):
    """Raise when download fails."""

    pass


class EmptyFilterError(IDPConfGenException):
    """Raise when PDB data filtering returns an empty selection."""

    errmsg = 'Filter returns empty selection, when saving file {}.'


class DSSPParserError(IDPConfGenException):
    """Raise when libs.libparse.DSSPParserError needs it."""

    errmsg = 'Error while parsing: {}'


class DSSPSecStructError(IDPConfGenException):
    """Raise when libs.libparse.DSSPParserError needs it."""

    errmsg = 'Values differ from possible DSSP secondary structure keys.'


class DSSPInputMissingError(IDPConfGenException):
    """Raise when libs.libparse.DSSPParserError needs it."""

    errmsg = 'One of the two required positional arguments is missing.'


class ParserNotFoundError(IDPConfGenException):
    """Raise when parser for PDB/CIF file is not found."""

    errmsg = 'Could not identity a proper parser.'


class NotBuiltError(IDPConfGenException):
    """Raise when attempting to access data of an object before building."""

    pass


class ReportOnCrashError(IDPConfGenException):
    """Raised when logger.report_on_crash."""

    errmsg = "Crash reported to {}."""
