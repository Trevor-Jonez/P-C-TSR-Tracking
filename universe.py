"""Coverage universe: US-listed P&C carriers with continuous trading history since 2020.

Segment classification is this project's own, based on each carrier's dominant business mix.
"""

UNIVERSE = [
    {"ticker": "PGR", "name": "Progressive", "segment": "Personal", "cik": 80661},
    {"ticker": "ALL", "name": "Allstate", "segment": "Personal", "cik": 899051},
    {"ticker": "MCY", "name": "Mercury General", "segment": "Personal", "cik": 64996},
    {"ticker": "KMPR", "name": "Kemper", "segment": "Personal", "cik": 860748},
    {"ticker": "TRV", "name": "Travelers", "segment": "Personal & commercial", "cik": 86312},
    {"ticker": "THG", "name": "Hanover Insurance", "segment": "Personal & commercial", "cik": 944695},
    {"ticker": "CB", "name": "Chubb", "segment": "Commercial (standard)", "cik": 896159},
    {"ticker": "AIG", "name": "AIG", "segment": "Commercial (standard)", "cik": 5272},
    {"ticker": "HIG", "name": "Hartford", "segment": "Commercial (standard)", "cik": 874766},
    {"ticker": "AFG", "name": "American Financial", "segment": "Commercial (standard)", "cik": 1042046},
    {"ticker": "CINF", "name": "Cincinnati Financial", "segment": "Commercial (standard)", "cik": 20286},
    {"ticker": "SIGI", "name": "Selective", "segment": "Commercial (standard)", "cik": 230557},
    {"ticker": "WRB", "name": "W. R. Berkley", "segment": "Commercial (specialty)", "cik": 11544},
    {"ticker": "ACGL", "name": "Arch Capital", "segment": "Commercial (specialty)", "cik": 947484},
    {"ticker": "AXS", "name": "AXIS Capital", "segment": "Commercial (specialty)", "cik": 1214816},
    {"ticker": "MKL", "name": "Markel", "segment": "Commercial (specialty)", "cik": 1096343},
    {"ticker": "RLI", "name": "RLI", "segment": "Commercial (specialty)", "cik": 84246},
    {"ticker": "KNSL", "name": "Kinsale", "segment": "Commercial (specialty)", "cik": 1669162},
]

BENCHMARKS = {"S&P 500": "^GSPC", "SPDR S&P Insurance ETF": "KIE"}
