"""Coverage universe with sub-industry and market lenses.

sub_industry : Carrier | Reinsurer | Retail broker | Wholesale & MGA | Insurtech
segment      : carrier segment used by the carrier analytics (None for non-carriers)
lens         : the market lens a company is most exposed to
market       : Admitted | Non-admitted (E&S) | Mixed | n/a (distribution / reinsurance)
Classifications are this project's own, based on each company's dominant business mix.
Nationwide, State Farm, Liberty Mutual, USAA and Farmers are mutual / privately held: no listed equity,
so they appear only inside industry aggregates.
"""

UNIVERSE = [
    # ---- Carriers: personal lines
    {"ticker": "PGR", "name": "Progressive", "sub_industry": "Carrier", "segment": "Personal", "lens": "Personal lines", "market": "Admitted", "cik": 80661},
    {"ticker": "ALL", "name": "Allstate", "sub_industry": "Carrier", "segment": "Personal", "lens": "Personal lines", "market": "Admitted", "cik": 899051},
    {"ticker": "MCY", "name": "Mercury General", "sub_industry": "Carrier", "segment": "Personal", "lens": "Personal lines", "market": "Admitted", "cik": 64996},
    {"ticker": "KMPR", "name": "Kemper", "sub_industry": "Carrier", "segment": "Personal", "lens": "Personal lines", "market": "Admitted", "cik": 860748},
    {"ticker": "HCI", "name": "HCI Group", "sub_industry": "Carrier", "segment": "Personal", "lens": "Personal lines (FL property)", "market": "Admitted", "cik": 1400810},
    {"ticker": "UVE", "name": "Universal Insurance", "sub_industry": "Carrier", "segment": "Personal", "lens": "Personal lines (FL property)", "market": "Admitted", "cik": 891166},
    # ---- Carriers: personal & commercial
    {"ticker": "TRV", "name": "Travelers", "sub_industry": "Carrier", "segment": "Personal & commercial", "lens": "Commercial", "market": "Admitted", "cik": 86312},
    {"ticker": "THG", "name": "The Hanover", "sub_industry": "Carrier", "segment": "Personal & commercial", "lens": "Commercial", "market": "Admitted", "cik": 944695},
    # ---- Carriers: commercial (standard)
    {"ticker": "CB", "name": "Chubb", "sub_industry": "Carrier", "segment": "Commercial (standard)", "lens": "Complex commercial", "market": "Mixed", "cik": 896159},
    {"ticker": "AIG", "name": "AIG", "sub_industry": "Carrier", "segment": "Commercial (standard)", "lens": "Complex commercial", "market": "Mixed", "cik": 5272},
    {"ticker": "HIG", "name": "The Hartford", "sub_industry": "Carrier", "segment": "Commercial (standard)", "lens": "Commercial", "market": "Admitted", "cik": 874766},
    {"ticker": "AFG", "name": "American Financial", "sub_industry": "Carrier", "segment": "Commercial (standard)", "lens": "Specialty", "market": "Mixed", "cik": 1042046},
    {"ticker": "CINF", "name": "Cincinnati Financial", "sub_industry": "Carrier", "segment": "Commercial (standard)", "lens": "Commercial", "market": "Admitted", "cik": 20286},
    {"ticker": "SIGI", "name": "Selective", "sub_industry": "Carrier", "segment": "Commercial (standard)", "lens": "Commercial", "market": "Admitted", "cik": 230557},
    {"ticker": "CNA", "name": "CNA Financial", "sub_industry": "Carrier", "segment": "Commercial (standard)", "lens": "Complex commercial", "market": "Mixed", "cik": 21175},
    # ---- Carriers: specialty / E&S
    {"ticker": "WRB", "name": "W. R. Berkley", "sub_industry": "Carrier", "segment": "Commercial (specialty)", "lens": "Specialty", "market": "Mixed", "cik": 11544},
    {"ticker": "ACGL", "name": "Arch Capital", "sub_industry": "Carrier", "segment": "Commercial (specialty)", "lens": "Specialty", "market": "Mixed", "cik": 947484},
    {"ticker": "AXS", "name": "AXIS Capital", "sub_industry": "Carrier", "segment": "Commercial (specialty)", "lens": "Specialty", "market": "Mixed", "cik": 1214816},
    {"ticker": "MKL", "name": "Markel", "sub_industry": "Carrier", "segment": "Commercial (specialty)", "lens": "Excess & surplus", "market": "Non-admitted (E&S)", "cik": 1096343},
    {"ticker": "RLI", "name": "RLI", "sub_industry": "Carrier", "segment": "Commercial (specialty)", "lens": "Excess & surplus", "market": "Mixed", "cik": 84246},
    {"ticker": "KNSL", "name": "Kinsale", "sub_industry": "Carrier", "segment": "Commercial (specialty)", "lens": "Excess & surplus", "market": "Non-admitted (E&S)", "cik": 1669162},
    {"ticker": "SKWD", "name": "Skyward Specialty", "sub_industry": "Carrier", "segment": "Commercial (specialty)", "lens": "Excess & surplus", "market": "Non-admitted (E&S)", "cik": 1519449},
    {"ticker": "BOW", "name": "Bowhead Specialty", "sub_industry": "Carrier", "segment": "Commercial (specialty)", "lens": "Excess & surplus", "market": "Non-admitted (E&S)", "cik": 2002473},
    {"ticker": "PLMR", "name": "Palomar", "sub_industry": "Carrier", "segment": "Commercial (specialty)", "lens": "Specialty", "market": "Mixed", "cik": 1761312},
    # ---- Reinsurers
    {"ticker": "RNR", "name": "RenaissanceRe", "sub_industry": "Reinsurer", "segment": None, "lens": "Reinsurance", "market": "n/a", "cik": 913144},
    {"ticker": "EG", "name": "Everest Group", "sub_industry": "Reinsurer", "segment": None, "lens": "Reinsurance", "market": "n/a", "cik": 1095073},
    {"ticker": "SPNT", "name": "SiriusPoint", "sub_industry": "Reinsurer", "segment": None, "lens": "Reinsurance", "market": "n/a", "cik": 1576018},
    {"ticker": "HG", "name": "Hamilton Insurance", "sub_industry": "Reinsurer", "segment": None, "lens": "Reinsurance", "market": "n/a", "cik": 1593275},
    # ---- Retail brokers
    {"ticker": "MRSH", "name": "Marsh McLennan", "sub_industry": "Retail broker", "segment": None, "lens": "Retail distribution", "market": "n/a", "cik": 62709},
    {"ticker": "AON", "name": "Aon", "sub_industry": "Retail broker", "segment": None, "lens": "Retail distribution", "market": "n/a", "cik": 315293},
    {"ticker": "AJG", "name": "Arthur J. Gallagher", "sub_industry": "Retail broker", "segment": None, "lens": "Retail distribution", "market": "n/a", "cik": 354190},
    {"ticker": "WTW", "name": "WTW", "sub_industry": "Retail broker", "segment": None, "lens": "Retail distribution", "market": "n/a", "cik": 1140536},
    {"ticker": "BRO", "name": "Brown & Brown", "sub_industry": "Retail broker", "segment": None, "lens": "Retail & wholesale distribution", "market": "n/a", "cik": 79282},
    {"ticker": "BWIN", "name": "Baldwin Group", "sub_industry": "Retail broker", "segment": None, "lens": "Retail distribution + MGA", "market": "n/a", "cik": 1781755},
    # ---- Wholesale & MGA / MGU platforms
    {"ticker": "RYAN", "name": "Ryan Specialty", "sub_industry": "Wholesale & MGA", "segment": None, "lens": "Wholesale / E&S + MGU", "market": "n/a", "cik": 1849253},
    {"ticker": "ARX", "name": "Accelerant", "sub_industry": "Wholesale & MGA", "segment": None, "lens": "MGA risk exchange", "market": "n/a", "cik": 1997350},
    {"ticker": "NP", "name": "Neptune Insurance", "sub_industry": "Wholesale & MGA", "segment": None, "lens": "MGA (flood)", "market": "n/a", "cik": 2067129},
    {"ticker": "HGTY", "name": "Hagerty", "sub_industry": "Wholesale & MGA", "segment": None, "lens": "MGA (specialty auto)", "market": "n/a", "cik": 1840776},
    # ---- Insurtech
    {"ticker": "ROOT", "name": "Root", "sub_industry": "Insurtech", "segment": None, "lens": "Personal lines (insurtech)", "market": "Admitted", "cik": 1788882},
    {"ticker": "LMND", "name": "Lemonade", "sub_industry": "Insurtech", "segment": None, "lens": "Personal lines (insurtech)", "market": "Admitted", "cik": 1691421},
]

# Listed abroad, no SEC filings: price data only (cyber specialists for the Cyber & AI page)
INTERNATIONAL = {"Hiscox (LSE)": "HSX.L"}  # Beazley delisted 2 Oct 2026 after Zurich takeover
BENCHMARKS = {"S&P 500": "^GSPC", "SPDR S&P Insurance ETF": "KIE"}
MUTUALS = ["State Farm", "Liberty Mutual", "Nationwide", "USAA", "Farmers"]

CARRIERS = [c for c in UNIVERSE if c["sub_industry"] == "Carrier"]
