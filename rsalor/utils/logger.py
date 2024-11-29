
# Logger -----------------------------------------------------------------------
class Logger:

    # Constants ----------------------------------------------------------------
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    # Constructor --------------------------------------------------------------
    def __init__(
            self,
            verbose: bool,
            disable_warnings: bool=True,
            step_prefix: str="STEP",
            warning_prefix: str="WARNING",
            error_prefix: str="ERROR",
            step_note: str="",
            warning_note: str="",
            error_note: str="",
        ):
        """Minimalistic logger:
            * manage verbose and disable_warnings
            * add colored prefixes to logs
        """
        self.verbose = verbose
        self.disable_warnings = disable_warnings
        self._step_prefix = step_prefix
        self._warning_prefix = warning_prefix
        self._error_prefix = error_prefix
        self._step_note = step_note
        self._warning_note = warning_note
        self._error_note = error_note

    # Methods ------------------------------------------------------------------
    @property
    def STEP_PREFIX(self) -> str:
        return f"{self.OKGREEN}{self._step_prefix}{self.ENDC}{self._step_note}"
    
    @property
    def WARNING_PREFIX(self) -> str:
        return f"{self.WARNING}{self._warning_prefix}{self.ENDC}{self._warning_note}"
    
    @property
    def CRITICAL_WARNING_PREFIX(self) -> str:
        return f"{self.FAIL}{self._warning_prefix}{self.ENDC}{self._warning_note}"
    
    @property
    def ERROR_PREFIX(self) -> str:
        return f"{self.FAIL}{self._error_prefix}{self.ENDC}{self._error_note}"

    def log(self, log_str: str) -> None:
        if self.verbose:
            print(log_str)

    def step(self, log_str: str) -> None:
        if self.verbose:
            print(f"{self.STEP_PREFIX}: {log_str}")

    def warning(self, log_str: str, critical: bool=False) -> None:
        prefix = self.WARNING_PREFIX
        if critical:
            prefix = self.CRITICAL_WARNING_PREFIX
        if not self.disable_warnings:
            print(f"{prefix}: {log_str}")

    def error(self, log_str: str) -> None:
        print(f"{self.ERROR_PREFIX}: {log_str}")

    def error_str(self, log_str: str) -> str:
        return f"{self.ERROR_PREFIX}: {log_str}"