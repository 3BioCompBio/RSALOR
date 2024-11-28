
# Imports ----------------------------------------------------------------------
import os.path
import csv
from typing import Union, Tuple, List, Dict, Callable
import numpy as np


# Main -------------------------------------------------------------------------
class CSV:
    """
    Class to read/write a CSV file and manage it as a dataframe.
        * It never assumes a column or a cell type except when it is specified (all cells as :str by default).
        * Manages safety: impossible to have redundent values in header.
    """

    # Constants ----------------------------------------------------------------
    ALLOWED_EXTENTIONS = ["csv", "tsv"]

    # Constructor --------------------------------------------------------------
    def __init__(
            self,
            header: List[str]=[],
            sep: str=";",
            name: str="DataFrame",
            print_warnings: bool=True,
        ):

        # Base properties
        self.name = name
        self.print_warnings = print_warnings

        # Content
        self._header = Header(header, sep)
        self.entries = []

    # Basic properties ---------------------------------------------------------
    @property
    def sep(self) -> str:
        return self._header.sep

    def __len__(self) -> int:
        return len(self.entries)
    
    def __contains__(self, property_name: str) -> bool:
        return property_name in self._header
    
    def __getitem__(self, id: int) -> dict:
        return self.entries[id]
    
    def __iter__(self):
        return iter(self.entries)
    
    @property
    def n_rows(self) -> int:
        return len(self)
    
    @property
    def n_cols(self) -> int:
        return len(self._header)
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_rows, self.n_cols)
    
    @property
    def df_size(self) -> int:
        return self.n_rows * self.n_cols
    
    def header(self) -> List[str]:
        return [p for p in self._header.properties]

    def __str__(self) -> str:
        return f"CSV('{self.name}', r={self.n_rows}, c={self.n_cols})"
    
    def warning(self, warning_str: str="") -> None:
        """Log a CSV Warning."""
        if self.print_warnings:
            print(f"WARNING in {self}{warning_str}")

    # Methods ------------------------------------------------------------------
    def set_sep(self, sep: str, safety_check: bool=True):
        """Set separator for the CSV (for .read and .write)"""

        if safety_check:
            
            # Computational time warning
            if self.df_size > 1000:
                self.warning(
                    f".set_sep('{sep}'): could be computationally expensive when CSV object already contains many entries. " + \
                    f"You can set 'safety_check' to False to skip coherence checks with separator."
                )

            # Guardians
            for entry in self.entries:
                for key, value in entry.items():
                    assert sep not in str(value), f"ERROR in {self}.set_sep('{sep}'): sep contained in entry's value ('{key}': '{value}')."
            
        # Set
        self._header.set_sep(sep)
        return self

    def add_entry(self, entry: dict):
        entry = {prop: entry[prop] for prop in self._header}
        self.entries.append(entry)
        return self

    def add_entries(self, entries: List[dict]):
        for entry in entries:
            self.add_entry(entry)
        return self

    def add_col(self, property: str, values: list, allow_replacement=False):
        if property in self._header:
            assert allow_replacement, f"ERROR in {self}.add_col(): property='{property}' already exists and allow_replacement is set to False."
        else:
            self._header.add(property)
        assert len(values) == len(self), f"ERROR in {self}.add_col(): values length ({len(values)}) != CSV length ({len(self)})."
        for entry, value in zip(self.entries, values):
            entry[property] = value
        return self
    
    def add_empty_col(self, property: str, missing_value: str="XXX", allow_replacement=False):
        values = [missing_value for _ in self.entries]
        self.add_col(property, values, allow_replacement=allow_replacement)
        return self
    
    def add_csv(self, other_csv):
        """Merge other_csv entries with current CSV (keeps header of current CSV)."""
        for property_name in self.header():
            assert property_name in other_csv._header, f"ERROR in {self}.add_csv(): property='{property_name}' does not exists in other_csv ({other_csv})."
        for entry in other_csv:
            self.add_entry(entry)
        return self

    def remove_col(self, property: str):
        self._header.remove(property)
        for entry in self.entries:
            del entry[property]
        return self

    def rename_col(self, property_old: str, property_new: str):
        self._header.rename(property_old, property_new)
        for entry in self.entries:
            entry[property_new] = entry[property_old]
            del entry[property_old]
        return self

    def order_header(self, header_order: List[str]):
        self._header.order(header_order)
        return self
    
    def filter(self, keep_entry_function: Callable, do_print: bool=False, filter_name: str=""):
        """Filter entries in the CSV with a filter_function."""
        l1 = len(self)
        self.entries = [entry for entry in self.entries if keep_entry_function(entry)]
        l2 = len(self)
        if do_print:
            print(f"{self}: Filter('{filter_name}'): {l1} -> {l2}")
        return self
    
    def set_col_type(self, property_name: str, dt: type, default_value=None):
        assert property_name in self._header, f"ERROR in {self}.set_col_type(): property_name='{property_name}' does not exists."
        for entry in self.entries:
            entry[property_name] = to_type(entry[property_name], dt, default_value=default_value)
    
    # Get Methods --------------------------------------------------------------
    def get_col(self, property: str, dt: Union[None, type]=None, default_value=None, as_numpy: bool=False):
        """Get Column of CSV as array."""
        assert property in self, f"ERROR in {self}.get_array('{property}'): property does not exists."
        col_list = [entry[property] for entry in self.entries]
        if dt is not None:
            col_list = [to_type(el, dt, default_value=default_value) for el in col_list]
        if as_numpy:
            col_list = np.array(col_list)
        return col_list
    
    def get_row(self, id: int, dt: Union[None, type]=None, default_value=None, as_numpy: bool=False):
        """Get Raw of CSV as array"""
        entry = self[id]
        row_list = [entry[p] for p in self._header]
        if dt is not None:
            row_list = [to_type(el, dt, default_value=default_value) for el in row_list]
        if as_numpy:
            row_list = np.array(row_list)
        return row_list
    
    def get_X(self, features: List[str]) -> np.ndarray:
        """Get features matrix X (numpy) from the CSV."""
        for feature in features:
            assert feature in self, f"ERROR in {self}.get_X(): feature='{feature}' does not exists."
        return np.array([
            [float(entry[feature]) for feature in features]
            for entry in self.entries
        ])
    
    def get_y(self, label: str) -> np.ndarray:
        """Get label array y (numpy) from the CSV."""
        assert label in self, f"ERROR in {self}.get_y(): label='{label}' does not exists."
        return np.array([float(entry[label]) for entry in self.entries])
    
    def get_Xy(self, features: List[str], label: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get (features, label) tuple (X, y) (numpy) from the CSV."""
        return self.get_X(features), self.get_y(label)
    
    @staticmethod
    def hash_entry(entry: dict, hash_properties: List[str], sep: str="_") -> str:
        """Hash an entry (to :str) by values of its hash_properties."""
        return sep.join([entry[prop] for prop in hash_properties])
    
    @staticmethod
    def get_hash_entry(hash_properties: List[str], sep: str="_") -> Callable:
        """Generate a hash_entry function."""
        def hash_entry_function(entry: dict) -> str:
            return sep.join([entry[prop] for prop in hash_properties])
        return hash_entry_function

    def get_map(self, hash_properties: List[str], sep: str="_", map_function: Union[None, Callable]=None) -> Dict[str, Dict]:
        """
        Obtain a map {hash(entry) -> entry} from CSV (redundencies not allowed).
            * if map_function is set, values of the map are defined as map_function(entry)
        """
        for property in hash_properties:
            assert property in self, f"ERROR in {self}.to_map(): property='{property}' not in header."
        entries_map = {}
        for entry in self.entries:
            h = self.hash_entry(entry, hash_properties, sep=sep)
            assert h not in entries_map, f"ERROR in {self}.to_map({hash_properties}) redundency found for '{h}'."
            entries_map[h] = entry
        if map_function is not None:
            for h, entry in entries_map.items():
                entry[h] = map_function(entry)
        return entries_map

    def get_groups(self, hash_properties: List[str], sep: str="_", map_function: Union[None, Callable]=None) -> Dict[str, List[Dict]]:
        """
        Obtain a map for groups {hash(entry) -> [entries_list]} from CSV.
            * if map_function is set, values of the map are defined as [map_function(entry), ...]
        """
        for property in hash_properties:
            assert property in self, f"ERROR in {self}.to_map(): property='{property}' not in header."
        groups_map = {}
        for entry in self:
            h = self.hash_entry(entry, hash_properties, sep=sep)
            if h not in groups_map:
                groups_map[h] = []
            groups_map[h].append(entry)
        if map_function is not None:
            for h, group in groups_map.items():
                groups_map[h] = [map_function(e) for e in group]
        return groups_map
    
    def copy(self):
        """Copy CSV object."""
        new_csv = CSV()
        new_csv.name = self.name
        new_csv.print_warnings = self.print_warnings
        new_csv._header = self._header.copy()
        new_csv.entries = [
            {k: v for k, v in entry.items()}
            for entry in self.entries
        ]
        return new_csv
    
    def show(self, n_entries: int=5, min_colsize: int=3, max_colsize: int=20, max_linesize: int=200, round_digit: int=4, sep: str=" | ") -> None:
        """Show summary of CSV."""
        lines = [self._header.properties] + [self.get_row(id) for id in range(min(n_entries, len(self)))]
        col_sizes = [
            max([min_colsize, min([max([len(stringify_float(line[i], round_digit=round_digit)) for line in lines]), max_colsize])])
            for i in range(len(self._header))
        ]
        print(self)
        for line in lines:
            print_line(line, sizes=col_sizes, max_linesize=max_linesize, round_digit=round_digit, sep=sep)
        if len(self) > n_entries:
            print("  ...")
    
    # IO -----------------------------------------------------------------------
    def write(self, output_path: str):
        """Save to file."""

        # Guardians
        output_path = os.path.abspath(output_path)
        assert any([output_path.endswith(f".{extention}")] for extention in CSV.ALLOWED_EXTENTIONS), f"ERROR in {self}.write('{output_path}'): extention sould be among {CSV.ALLOWED_EXTENTIONS})."
        assert os.path.isdir(os.path.dirname(output_path)), f"ERROR in {self}.write('{output_path}'): destination folder does not exists."
        if output_path.endswith("tsv"):
            assert self.sep == "\t",  f"ERROR in {self}.write('{output_path}'): if extention is '.tsv', separator should be '\\t' however sep='{self.sep}'."
        
        # Stringify
        str_header = self.sep.join(self._header.properties)
        str_entries_list = [
            self.sep.join(str(entry[prop]) for prop in self._header.properties)
            for entry in self.entries
        ]
        str_lines = [str_header] + str_entries_list

        # Write
        with open(output_path, "w") as fs:
            fs.write("\n".join(str_lines))
        return self
    
    def read(self, input_path: str, col_types: Dict[str, type]={}, col_default: dict={}):
        """Read from file."""

        # Guardians
        assert any([input_path.endswith(f".{extention}")] for extention in CSV.ALLOWED_EXTENTIONS), f"ERROR in {self}.read('{input_path}'): extention sould be among {CSV.ALLOWED_EXTENTIONS})."
        assert os.path.isfile(input_path), f"ERROR in {self}.read('{input_path}'): input_path file does not exists."

        # Set name
        file_name = os.path.basename(input_path)
        name = ".".join(file_name.split(".")[:-1])
        self.name = name

        # Parse csv from file
        with open(input_path, newline='') as csvfile:
            csv_lines = list(csv.reader(csvfile, delimiter=self.sep))

        # Set CSV header
        header = csv_lines[0]
        if len(header) <= 1:
            self.warning(f".read('{input_path}'): header contains {len(header)} values. Maybe sep='{self.sep}' parameter in incorrect.")
        self._header = Header(header, self.sep)

        # Set CSV entries
        self.entries = []
        for i, line in enumerate(csv_lines[1:]):
            assert len(line) == len(header), f"ERROR in {self}.read('{input_path}'): number of elements ({len(line)}) in entry ({i+1}/{len(csv_lines)-1}) does not match the header ({len(header)})."
            self.entries.append({prop: value for prop, value in zip(header, line)})

        # Set column types if required
        for col_name, dt in col_types.items():
            col_default_value = col_default.get(col_name, None)
            self.set_col_type(col_name, dt, default_value=col_default_value)

        return self

# Dependencies -----------------------------------------------------------------

class Header:
    """
    Container for the Header of a CSV object.
        -> ordered list with no repetitions allowed and a separator of length = 1.
    """

    # Constructor --------------------------------------------------------------
    def __init__(self, properties: List[str], sep: str):

        # Init
        self.sep = ""
        self.properties = []
        self.properties_set = set()

        # Set header values
        self.set_sep(sep)
        for property in properties:
            self.add(property)

    # Basic properties ---------------------------------------------------------
    def __getitem__(self, id: int) -> str:
        return self.properties[id]

    def __iter__(self):
        return iter(self.properties)
    
    def __contains__(self, property_name: str) -> bool:
        return property_name in self.properties_set

    def __len__(self) -> int:
        return len(self.properties)
    
    def __str__(self) -> str:
        return f"CSV.Header(l={len(self)})"

    def show(self):
        MAX_CHAR = 80
        properties_str = f"'{self.properties[0]}'"
        for property in self.properties[1:]:
            if len(properties_str) + len(property) > MAX_CHAR:
                properties_str += ", ..."
                break
            properties_str += f", '{property}'"
        print(f"CSV.Header([{properties_str}], len={len(self)}, sep='{self.sep}')")
        return self
    
    def idof(self, property_name: str) -> int:
        assert property_name in self, f"ERROR in {self}.idof(): property_name='{property_name}' not in header."
        for i, current_property_name in enumerate(self):
            if property_name == current_property_name:
                return i
    
    # Methods ------------------------------------------------------------------
    def set_sep(self, sep: str):
        assert len(sep) == 1, f"ERROR in {self}.set_sep(): sep='{sep}' should be of length 1."
        for property in self:
            assert sep not in property, f"ERROR in {self}.set_sep(): sep='{sep}' is contained in property '{property}'."        
        self.sep = sep
        return self

    def add(self, property_name: str):
        assert self.sep not in property_name, f"ERROR in {self}.add('{property_name}'): property contains sep='{self.sep}'."
        assert property_name not in self, f"ERROR in {self}.add('{property_name}'): property already exists."
        self.properties.append(property_name)
        self.properties_set.add(property_name)
        return self

    def remove(self, property_name: str):
        assert property_name in self, f"ERROR in {self}.remove('{property_name}'): property does not exists."
        self.properties.remove(property_name)
        self.properties_set.remove(property_name)
        return self

    def rename(self, property_old: str, property_new: str):
        assert property_old != property_new, f"ERROR in {self}.rename(): old property and new property have the same value '{property_old}'."
        assert property_old in self, f"ERROR in {self}.rename(): old property '{property_old}' is not in header."
        assert property_new not in self, f"ERROR in {self}.rename(): new property '{property_new}' already in header."
        assert self.sep not in property_new, f"ERROR in {self}.rename(): new property '{property_new}' contains sep='{self.sep}'."
        id = self.idof(property_old)
        self.properties[id] = property_new
        self.properties_set.add(property_new)
        self.properties_set.remove(property_old)
        return self

    def order(self, header_order: List[str]):
        for property in header_order:
            assert property in self, f"ERROR in {self}.order(): property '{property}' not in header."
        ordered_properties_set = set(header_order)
        unordered_properties = [property for property in self if property not in ordered_properties_set]
        self.properties = header_order + unordered_properties
        return self
    
    def copy(self):
        return Header([p for p in self], self.sep)
    
# Dependency: Utils Funcions ---------------------------------------------------
def to_type(input, dt:type, default_value=None):
    """Convert input to type dt. If default_value is set, returns default_value when convertion fails."""
    try:
        return dt(input)
    except:
        if default_value is None:
            raise ValueError(f"ERROR in CSV().to_type(): input='{input}' not convertable to {dt}. Please correct input or set a default_value.")
        else:
            return default_value

def print_line(
        line_list,
        sep: str=" | ", dots_str: str="...",
        size: int=20, sizes: Union[None, List[int]]=None, max_linesize: int=200,
        round_digit: int=4,
    ) -> None:
    """Print a line from a table (dataframe) in a standardized way."""
    if sizes is None: sizes = [size for _ in line_list]
    line_str = ""
    unprinted_cols = False
    for element, size in zip(line_list, sizes):
        line_new_col = sep + format_string(element, size, round_digit=round_digit)
        if len(line_str) + len(line_new_col) > max_linesize - (len(sep) + len(dots_str)):
            unprinted_cols = True
            break
        line_str += line_new_col
    if unprinted_cols:
        line_str += sep + "..."
    line_str += sep
    print(line_str[1:-1])

def format_string(input, size: int=20, filler: int=" ", dots_str: str="...", round_digit: int=4) -> str:
    """Format a string to standardized form (length, ...)"""
    input_str = stringify_float(input, round_digit=round_digit)
    if len(input_str) > size:
        return input_str[:size-len(dots_str)] + dots_str
    else:
        return input_str + filler*(size - len(input_str))
    
def stringify_float(input, round_digit: int=4) -> str:
    if isinstance(input, float):
        str_float = str(round(input, round_digit))
        n_digits = len(str_float.split(".")[-1])
        str_float  = str_float + ("0"*(round_digit-n_digits))
        if str_float[0] != "-":
            str_float = " " + str_float
        return str_float
    else:
        return str(input)
