
# Imports ----------------------------------------------------------------------
from typing import List, Dict

# AminoAcid --------------------------------------------------------------------
class AminoAcid:
    """Container class for the 20 standard proteogenic amino acids.
        * Manages mapping between amino acids: id, one-letter-code and three-letter-code
        * Manages non-standard amino acids three-letters-codes and their corresponding standard AA

    usage:
    
    alanine_id: int = AminoAcid.ONE_2_ID['A'] \n
    one_letter_code_3: str = AminoAcid.ID_2_ONE[3] \n
    ala: AminoAcid= AminoAcid('A') \n
    non_standard_metionine: AminoAcid = AminoAcid.parse_three('MSE')
    """

    # Static Properties --------------------------------------------------------

    # Standard Amino Acids metadata
    _AA_LIST = [
        ( 0, "A", "ALA", "alanine"),
        ( 1, "C", "CYS", "cysteine"),
        ( 2, "D", "ASP", "aspartate"),
        ( 3, "E", "GLU", "glutamate"),
        ( 4, "F", "PHE", "phenylalanine"),
        ( 5, "G", "GLY", "glycine"),
        ( 6, "H", "HIS", "histidine"),
        ( 7, "I", "ILE", "isoleucine"),
        ( 8, "K", "LYS", "lysine"),
        ( 9, "L", "LEU", "leucine"),
        (10, "M", "MET", "methionine"),
        (11, "N", "ASN", "asparagine"),
        (12, "P", "PRO", "proline"),
        (13, "Q", "GLN", "glutamine"),
        (14, "R", "ARG", "arginine"),
        (15, "S", "SER", "serine"),
        (16, "T", "THR", "thréonine"),
        (17, "V", "VAL", "valine"),
        (18, "W", "TRP", "tryptophane"),
        (19, "Y", "TYR", "tyrosine"),
    ]

    # Gap and Unknown Amino Acid properties
    GAP_ID = 20
    GAP_ONE = "-"
    GAP_THREE = "GAP"
    GAP_NAME = "gap"
    UNK_ID = -1
    UNK_ONE = "X"
    UNK_THREE = "XXX"
    UNK_NAME = "unknown"

    # Non-Standard Amino Acids three-letter-codes mapped to closest Standard Amino Acids
    _NON_STANDARD_AAS = {
        # WARNING: We do not represent here ambiguous mappings like:   "GLX" => "GLN" or "GLU"
        "4HT": "TRP", "CLG": "LYS", "HSE": "SER", "BIF": "PHE", "B3D": "ASP", "BB8": "PHE", "3MY": "TYR", "SNK": "HIS",
        "3CF": "PHE", "A5N": "ASN", "LED": "LEU", "TOX": "TRP", "CR5": "GLY", "ILM": "ILE", "0A9": "PHE", "DAS": "ASP",
        "NYS": "CYS", "73P": "LYS", "MSO": "MET", "IYR": "TYR", "PR9": "PRO", "R4K": "TRP", "L5P": "LYS", "31Q": "CYS",
        "OCY": "CYS", "BH2": "ASP", "XSN": "ASN", "SXE": "SER", "GMA": "GLU", "SEP": "SER", "CYD": "CYS", "YPZ": "TYR",
        "GPL": "LYS", "RVX": "SER", "YCM": "CYS", "SEL": "SER", "DNE": "LEU", "LEN": "LEU", "4FB": "PRO", "4OU": "PHE",
        "LGY": "LYS", "TTQ": "TRP", "DBB": "THR", "LBZ": "LYS", "QX7": "ALA", "H14": "PHE", "CIR": "ARG", "73O": "TYR",
        "EI4": "ARG", "LVN": "VAL", "SRZ": "SER", "55I": "PHE", "UF0": "SER", "YHA": "LYS", "QM8": "ALA", "TQQ": "TRP",
        "QIL": "ILE", "Q75": "MET", "11Q": "PRO", "A8E": "VAL", "DHV": "VAL", "3BY": "PRO", "2ZC": "SER", "T9E": "THR",
        "CSZ": "CYS", "5CS": "CYS", "KPI": "LYS", "0AH": "SER", "HSK": "HIS", "TH6": "THR", "ARO": "ARG", "E9V": "HIS",
        "UXQ": "PHE", "MHL": "LEU", "CAS": "CYS", "8RE": "LYS", "LLP": "LYS", "PTH": "TYR", "ORQ": "ARG", "73N": "ARG",
        "BTK": "LYS", "HVA": "VAL", "LMQ": "GLN", "FME": "MET", "XX1": "LYS", "I7F": "SER", "4N9": "PRO", "TYJ": "TYR",
        "BOR": "ARG", "HL2": "LEU", "73C": "SER", "0CS": "ALA", "AGM": "ARG", "CYW": "CYS", "ASL": "ASP", "I3D": "TRP",
        "NPH": "CYS", "JKH": "PRO", "QMB": "ALA", "XCN": "CYS", "PHI": "PHE", "NAL": "ALA", "LYZ": "LYS", "6M6": "CYS",
        "VAD": "VAL", "EXL": "TRP", "WFP": "PHE", "823": "ASN", "CLH": "LYS", "C6C": "CYS", "DCY": "CYS", "DPP": "ALA",
        "KHB": "LYS", "DNW": "ALA", "BUC": "CYS", "CSU": "CYS", "H5M": "PRO", "RXL": "VAL", "FOE": "CYS", "GHP": "GLY",
        "2KP": "LYS", "OMX": "TYR", "ZCL": "PHE", "MGG": "ARG", "DLS": "LYS", "30V": "CYS", "02K": "ALA", "DA2": "ARG",
        "TYY": "TYR", "HRG": "ARG", "PHL": "PHE", "PRJ": "PRO", "M2L": "LYS", "SUN": "SER", "TSY": "CYS", "PF5": "PHE",
        "4CF": "PHE", "1OP": "TYR", "CSB": "CYS", "POM": "PRO", "ELY": "LYS", "TRQ": "TRP", "BP5": "ALA", "5VV": "ASN",
        "6DN": "LYS", "MIS": "SER", "MLZ": "LYS", "EME": "GLU", "4J5": "ARG", "MPQ": "GLY", "LLO": "LYS", "FQA": "LYS",
        "PR7": "PRO", "NLW": "LEU", "OMY": "TYR", "5CT": "LYS", "PRK": "LYS", "DPQ": "TYR", "N0A": "PHE", "3QN": "LYS",
        "K5H": "CYS", "HNC": "CYS", "TYO": "TYR", "Q3P": "LYS", "BWV": "ARG", "4L0": "PRO", "ZAL": "ALA", "IAM": "ALA",
        "AGQ": "TYR", "07O": "CYS", "PCA": "GLN", "2MR": "ARG", "TRN": "TRP", "4AR": "ARG", "HLY": "LYS", "DHI": "HIS",
        "J2F": "TYR", "C3Y": "CYS", "GL3": "GLY", "BTR": "TRP", "OYL": "HIS", "IGL": "GLY", "2GX": "PHE", "8LJ": "PRO",
        "AYA": "ALA", "XYC": "ALA", "CY1": "CYS", "CGU": "GLU", "PM3": "PHE", "03Y": "CYS", "CE7": "ASN", "HSL": "SER",
        "BXT": "SER", "MHU": "PHE", "HOX": "PHE", "5GM": "ILE", "DVA": "VAL", "CYR": "CYS", "YOF": "TYR", "DDZ": "ALA",
        "4PQ": "TRP", "ECC": "GLN", "GHG": "GLN", "IPG": "GLY", "PPN": "PHE", "L3O": "LEU", "AEA": "CYS", "7N8": "PHE",
        "AHO": "ALA", "TBG": "VAL", "BFD": "ASP", "HPE": "PHE", "5MW": "LYS", "U2X": "TYR", "N10": "SER", "TGH": "TRP",
        "51T": "TYR", "DDE": "HIS", "DBZ": "ALA", "FF9": "LYS", "HTN": "ASN", "NVA": "VAL", "HS9": "HIS", "ACB": "ASP",
        "9KP": "LYS", "FTR": "TRP", "ALS": "ALA", "DYJ": "PRO", "RPI": "ARG", "FTY": "TYR", "TQZ": "CYS", "FVA": "VAL",
        "CS4": "CYS", "QVA": "CYS", "XPR": "PRO", "0QL": "CYS", "TCQ": "TYR", "OXX": "ASP", "ZZJ": "ALA", "LDH": "LYS",
        "3CT": "TYR", "H7V": "ALA", "4N7": "PRO", "PYA": "ALA", "WVL": "VAL", "DMK": "ASP", "EFC": "CYS", "0BN": "PHE",
        "MHO": "MET", "ECX": "CYS", "ESB": "TYR", "KGC": "LYS", "3WX": "PRO", "MBQ": "TYR", "ILX": "ILE", "DSG": "ASN",
        "P2Q": "TYR", "LSO": "LYS", "6CW": "TRP", "SDP": "SER", "MP8": "PRO", "HTR": "TRP", "B3S": "SER", "TYB": "TYR",
        "PAQ": "TYR", "HS8": "HIS", "RX9": "ILE", "DHA": "SER", "CHP": "GLY", "MMO": "ARG", "FCL": "PHE", "05O": "TYR",
        "ICY": "CYS", "DIV": "VAL", "N65": "LYS", "Q78": "PHE", "KCR": "LYS", "TY8": "TYR", "GVL": "SER", "MLL": "LEU",
        "DNP": "ALA", "5XU": "ALA", "O7D": "TRP", "NFA": "PHE", "DBY": "TYR", "QCS": "CYS", "ZYK": "PRO", "IIL": "ILE",
        "ABA": "ALA", "4AW": "TRP", "BSE": "SER", "LLY": "LYS", "4D4": "ARG", "MNL": "LEU", "FGL": "GLY", "SET": "SER",
        "MYN": "ARG", "C4R": "CYS", "CZZ": "CYS", "CZS": "ALA", "Y1V": "LEU", "CWR": "SER", "NBQ": "TYR", "KYQ": "LYS",
        "2TY": "TYR", "1PA": "PHE", "6V1": "CYS", "FGP": "SER", "BB9": "CYS", "AGT": "CYS", "CYG": "CYS", "VI3": "CYS",
        "PH6": "PRO", "NZH": "HIS", "DAB": "ALA", "B2A": "ALA", "6WK": "CYS", "PR4": "PRO", "7O5": "ALA", "OHS": "ASP",
        "3YM": "TYR", "Z3E": "THR", "NC1": "SER", "CAF": "CYS", "BPE": "CYS", "BB7": "CYS", "RE0": "TRP", "TSQ": "PHE",
        "4CY": "MET", "G5G": "LEU", "TDD": "LEU", "KCX": "LYS", "0AR": "ARG", "HSV": "HIS", "2ML": "LEU", "4PH": "PHE",
        "V44": "CYS", "IAS": "ASP", "FH7": "LYS", "PTM": "TYR", "SAR": "GLY", "SVX": "SER", "MEN": "ASN", "CS1": "CYS",
        "HOO": "HIS", "NYB": "CYS", "HMR": "ARG", "05N": "PRO", "V61": "PHE", "41H": "PHE", "BMT": "THR", "4HL": "TYR",
        "I2M": "ILE", "4N8": "PRO", "2RX": "SER", "CS3": "CYS", "MEA": "PHE", "B2F": "PHE", "CYF": "CYS", "GNC": "GLN",
        "4HJ": "SER", "CSJ": "CYS", "2SO": "HIS", "Q2E": "TRP", "CXM": "MET", "4WQ": "ALA", "5OW": "LYS", "TRX": "TRP",
        "B3Y": "TYR", "DAH": "PHE", "5PG": "GLY", "ESC": "MET", "DTY": "TYR", "CGA": "GLU", "TFW": "TRP", "SMF": "PHE",
        "S1H": "SER", "SAC": "SER", "QCI": "GLN", "CMT": "CYS", "TY2": "TYR", "0A8": "CYS", "OMH": "SER", "QPA": "CYS",
        "MK8": "LEU", "DLE": "LEU", "T0I": "TYR", "ALT": "ALA", "3X9": "CYS", "5CW": "TRP", "9E7": "LYS", "MGN": "GLN",
        "PBF": "PHE", "AEI": "THR", "TYI": "TYR", "SNN": "ASN", "74P": "LYS", "OHI": "HIS", "KST": "LYS", "SBL": "SER",
        "JJJ": "CYS", "JJL": "CYS", "2RA": "ALA", "DIL": "ILE", "02Y": "ALA", "CYJ": "LYS", "2HF": "HIS", "FC0": "PHE",
        "NLN": "LEU", "XW1": "ALA", "QMM": "GLN", "TOQ": "TRP", "WPA": "PHE", "TIH": "ALA", "NLB": "LEU", "BG1": "SER",
        "PTR": "TYR", "0WZ": "TYR", "ZYJ": "PRO", "SNC": "CYS", "BBC": "CYS", "B3E": "GLU", "4GJ": "CYS", "MSA": "GLY",
        "TPO": "THR", "HIQ": "HIS", "PHA": "PHE", "THC": "THR", "JJK": "CYS", "API": "LYS", "TY5": "TYR", "LPD": "PRO",
        "MND": "ASN", "PRV": "GLY", "M3L": "LYS", "HR7": "ARG", "86N": "GLU", "DSN": "SER", "5R5": "SER", "IC0": "GLY",
        "ARM": "ARG", "4AK": "LYS", "HT7": "TRP", "E9M": "TRP", "4DP": "TRP", "IML": "ILE", "BCS": "CYS", "7OZ": "ALA",
        "2MT": "PRO", "GLZ": "GLY", "0E5": "THR", "U3X": "PHE", "HYP": "PRO", "M0H": "CYS", "7XC": "PHE", "AZK": "LYS",
        "AHB": "ASN", "NCB": "ALA", "ASA": "ASP", "TPL": "TRP", "0TD": "ASP", "HTI": "CYS", "LRK": "LYS", "ME0": "MET",
        "143": "CYS", "FY2": "TYR", "1TY": "TYR", "QPH": "PHE", "F2F": "PHE", "3PX": "PRO", "PLJ": "PRO", "N9P": "ALA",
        "3ZH": "HIS", "C5C": "CYS", "PFF": "PHE", "NEP": "HIS", "CSA": "CYS", "4J4": "CYS", "O7G": "VAL", "TTS": "TYR",
        "KFP": "LYS", "FZN": "LYS", "TYN": "TYR", "AA4": "ALA", "LYX": "LYS", "HP9": "PHE", "TH5": "THR", "D2T": "ASP",
        "MED": "MET", "TRW": "TRP", "HLU": "LEU", "CSO": "CYS", "23F": "PHE", "PG9": "GLY", "EJA": "CYS", "RE3": "TRP",
        "66D": "ILE", "4OG": "TRP", "MSE": "MET", "MDF": "TYR", "DBU": "THR", "SEN": "SER", "Y57": "LYS", "XA6": "PHE",
        "M2S": "MET", "FLT": "TYR", "GME": "GLU", "LE1": "VAL", "FY3": "TYR", "OZW": "PHE", "FP9": "PRO", "FHL": "LYS",
        "MLE": "LEU", "DAR": "ARG", "BHD": "ASP", "LA2": "LYS", "SLZ": "LYS", "CSX": "CYS", "OCS": "CYS", "DMH": "ASN",
        "2CO": "CYS", "NLE": "LEU", "LME": "GLU", "HIC": "HIS", "ZBZ": "CYS", "MYK": "LYS", "2JG": "SER", "ORN": "ALA",
        "YTF": "GLN", "1AC": "ALA", "OLD": "HIS", "B2I": "ILE", "HZP": "PRO", "4AF": "PHE", "OMT": "MET", "CSP": "CYS",
        "APK": "LYS", "DPR": "PRO", "CY0": "CYS", "5T3": "LYS", "CY3": "CYS", "3GL": "GLU", "4II": "PHE", "0AK": "ASP",
        "ALC": "ALA", "LP6": "LYS", "HIP": "HIS", "60F": "CYS", "CML": "CYS", "CYQ": "CYS", "NA8": "ALA", "MH6": "SER",
        "GFT": "SER", "WLU": "LEU", "AZH": "ALA", "KBE": "LYS", "LCK": "LYS", "LAY": "LEU", "0LF": "PRO", "KKD": "ASP",
        "K7K": "SER", "CSR": "CYS", "B3K": "LYS", "OSE": "SER", "F2Y": "TYR", "NMM": "ARG", "P1L": "CYS", "PRS": "PRO",
        "OBS": "LYS", "ZDJ": "TYR", "BYR": "TYR", "HY3": "PRO", "ASB": "ASP", "NLY": "GLY", "0A1": "TYR", "DPL": "PRO",
        "SCS": "CYS", "I4G": "GLY", "6CV": "ALA", "HIA": "HIS", "LYN": "LYS", "54C": "TRP", "FGA": "GLU", "B27": "THR",
        "TYE": "TYR", "DTH": "THR", "PSH": "HIS", "EXA": "LYS", "BLE": "LEU", "P9S": "CYS", "23P": "ALA", "1TQ": "TRP",
        "RVJ": "ALA", "ALO": "THR", "FL6": "ASP", "4LZ": "TYR", "TMD": "THR", "FHO": "LYS", "0FL": "ALA", "AN6": "LEU",
        "4OV": "SER", "432": "SER", "SCH": "CYS", "DGL": "GLU", "2TL": "THR", "TPQ": "TYR", "3AH": "HIS", "CSD": "CYS",
        "PR3": "CYS", "IZO": "MET", "DV9": "GLU", "41Q": "ASN", "DI7": "TYR", "34E": "VAL", "MHS": "HIS", "GGL": "GLU",
        "ALY": "LYS", "O6H": "TRP", "8JB": "CYS", "SVV": "SER", "KOR": "MET", "PYX": "CYS", "6CL": "LYS", "WRP": "TRP",
        "SCY": "CYS", "G1X": "TYR", "2KK": "LYS", "TYQ": "TYR", "MIR": "SER", "ALN": "ALA", "CMH": "CYS", "KPY": "LYS",
        "SVZ": "SER", "NMC": "GLY", "RGL": "ARG", "SME": "MET", "DAL": "ALA", "DTR": "TRP", "PEC": "CYS", "SGB": "SER",
        "NLO": "LEU", "AHP": "ALA", "SLL": "LYS", "TRF": "TRP", "CME": "CYS", "SEE": "SER", "MME": "MET", "DYA": "ASP",
        "33X": "ALA", "LYF": "LYS", "CZ2": "CYS", "TRO": "TRP", "DPN": "PHE", "IB9": "TYR", "POK": "ARG", "LET": "LYS",
        "CCS": "CYS", "DGN": "GLN", "NIY": "TYR", "E9C": "TYR", "SEB": "SER", "AIB": "ALA", "OAS": "SER", "V7T": "LYS",
        "K5L": "SER", "TYS": "TYR", "FIO": "ARG", "B2V": "VAL", "GLJ": "GLU", "JLP": "LYS", "MVA": "VAL", "0Y8": "PRO",
        "OTH": "THR", "00C": "CYS", "0EA": "TYR", "F7W": "TRP", "LEI": "VAL", "UMA": "ALA", "OLT": "THR", "4KY": "PRO",
        "MCS": "CYS", "TNQ": "TRP", "HIX": "ALA", "C1X": "LYS", "PAT": "TRP", "T8L": "THR", "DM0": "LYS", "CG6": "CYS",
        "KPF": "LYS", "DYS": "CYS", "BB6": "CYS", "LAL": "ALA", "DLY": "LYS", "DJD": "PHE", "LTU": "TRP", "TYT": "TYR",
        "VPV": "LYS", "D11": "THR", "LEF": "LEU", "1X6": "SER", "ML3": "LYS", "MAA": "ALA", "7ID": "ASP", "AAR": "ARG",
        "NZC": "THR", "R1A": "CYS", "CGV": "CYS", "D3P": "GLY", "TIS": "SER", "LYR": "LYS", "4IN": "TRP", "CY4": "CYS",
        "0AF": "TRP", "TLY": "LYS", "SVA": "SER", "4HH": "SER", "HQA": "ALA", "PHD": "ASP", "KYN": "TRP", "4FW": "TRP",
        "VHF": "GLU", "CTH": "THR", "B3X": "ASN", "MTY": "TYR", "MLY": "LYS", "SMC": "CYS", "TS9": "ILE", "PXU": "PRO",
        "DSE": "SER", "P3Q": "TYR", "BCX": "CYS", "FAK": "LYS", "SVY": "SER", "CSS": "CYS", "FDL": "LYS", "2LT": "TYR",
        "N80": "PRO", "B3A": "ALA", "LYO": "LYS", "VR0": "ARG", "YTH": "THR",
    }

    # Mappings to aa metadata
    _ID_MAP: Dict[int, tuple] = {aa[0]: aa for aa in _AA_LIST}
    _ONE_MAP: Dict[str, tuple] = {aa[1]: aa for aa in _AA_LIST}
    _THREE_MAP: Dict[str, tuple] = {aa[2]: aa for aa in _AA_LIST}
    #_AA_MAP = _ID_MAP | _ONE_MAP | _THREE_MAP

    # Translation tables
    THREE_2_ONE: Dict[str, str] = {aa[2]: aa[1] for aa in _AA_LIST}
    ONE_2_THREE: Dict[str, str] = {aa[1]: aa[2] for aa in _AA_LIST}
    ONE_2_ID: Dict[str, int] = {aa[1]: aa[0] for aa in _AA_LIST}
    ID_2_ONE: Dict[int, str] = {aa[0]: aa[1] for aa in _AA_LIST}

    # Construcors --------------------------------------------------------------
    def __init__(self, aa_one: str) -> "AminoAcid":
        """Only accepts standard Amino Acid one-letter-codes."""
        assert aa_one in self._ONE_MAP, f"ERROR in AminoAcid('{aa_one}'): invalid amino acid one-letter-code."
        aa_metadata = self._ONE_MAP[aa_one]
        self.id: int = aa_metadata[0]
        self.one: str = aa_metadata[1]
        self.three: str = aa_metadata[2]
        self.three_standard: str = aa_metadata[2]
        self.name: str = aa_metadata[3]

    @classmethod
    def parse_three(cls, aa_three: int) -> "AminoAcid":
        """Parse an AminoAcid from its three-letter-code (can handle non-standard AAs and mapping to corresponding standard AA)."""

        # Standard case
        aa_one = cls.THREE_2_ONE.get(aa_three, None)
        if aa_one is not None:
            return AminoAcid(aa_one)
        
        # Non-standard case
        aa_three_standard = cls._NON_STANDARD_AAS.get(aa_three, None)
        if aa_three_standard is not None:
            aa_one = cls.THREE_2_ONE[aa_three_standard]
            aa = AminoAcid(aa_one)
            aa.three = aa_three
            return aa
        
        # Unknown case
        return cls.get_unknown()

    @classmethod
    def get_all(cls) -> List["AminoAcid"]:
        """Get the list of all 20 standard AminoAcids."""
        return [AminoAcid(aa_metadata[1]) for aa_metadata in cls._AA_LIST]
    
    @classmethod
    def get_unknown(cls) -> "AminoAcid":
        """Return an unknown AminoAcid."""
        aa = AminoAcid("A")
        aa.id = AminoAcid.UNK_ID
        aa.one = AminoAcid.UNK_ONE
        aa.three = AminoAcid.UNK_THREE
        aa.three_standard = None
        aa.name = AminoAcid.UNK_NAME
        return aa
    
    @classmethod
    def get_gap(cls) -> "AminoAcid":
        """Return a gap 'AminoAcid'."""
        aa = AminoAcid("A")
        aa.id = AminoAcid.GAP_ID
        aa.one = AminoAcid.GAP_ONE
        aa.three = AminoAcid.GAP_THREE
        aa.three_standard = None
        aa.name = AminoAcid.GAP_NAME
        return aa

    # Base properties ----------------------------------------------------------
    def __str__(self) -> str:
        if self.is_standard:
            return f"AminoAcid('{self.one}', '{self.three}', id={self.id})"
        else:
            return f"AminoAcid('{self.one}', '{self.three}' (std='{self.three_standard}'), id={self.id})"
        
    def is_gap(self) -> bool:
        self.id == AminoAcid.GAP_ID

    def is_unknown(self) -> bool:
        return self.id == AminoAcid.UNK_ID
    
    def is_aminoacid(self) -> bool:
        return not self.is_gap()
    
    def is_standard(self) -> bool:
        return self.three == self.three_standard

    # Class methods ------------------------------------------------------------
    @classmethod
    def id_exists(cls, id: int) -> bool:
        """Return if 'id' corresponds to the id of a standard Amino Acid."""
        return id in cls._ID_MAP

    @classmethod
    def one_exists(cls, aa_one: str) -> bool:
        """Return if 'aa_one' corresponds to the one-letter-code of a standard Amino Acid."""
        return aa_one in cls._ONE_MAP
    
    @classmethod
    def three_exists(cls, aa_three: str) -> bool:
        """Return if 'aa_three' corresponds to the three-letter-code of a standard Amino Acid."""
        return aa_three in cls._THREE_MAP
