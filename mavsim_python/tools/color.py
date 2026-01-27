import os

if os.name == 'nt':
    os.system('color')
#

ENDC    = '\033[0m'
VIOLET  = '\033[95m'
BLUE    = '\033[94m'
CYAN    = '\033[96m'
GREEN   = '\033[92m'
# YELLOW  = '\033[93m'
YELLOW  = '\033[33m'
RED     = '\033[91m'
# REDBG   = '\033[41m'
REDBG   = '\033[101m'
BOLD    = '\033[1m'
ULINE   = '\033[4m'


def violet(ss):
    return f"{VIOLET}{ss}{ENDC}"
#
def blue(ss):
    return f"{BLUE}{ss}{ENDC}"
#
def cyan(ss):
    return f"{CYAN}{ss}{ENDC}"
#
def green(ss):
    return f"{GREEN}{ss}{ENDC}"
#
def yellow(ss):
    return f"{YELLOW}{ss}{ENDC}"
#
def red(ss):
    return f"{RED}{ss}{ENDC}"
#
def redbg(ss):
    return f"{REDBG}{ss}{ENDC}"
#
def uline(ss):
    return f"{ULINE}{ss}{ENDC}"
#

# fg = lambda text, color: "\33[38;5;" + str(color) + "m" + text + "\33[0m"
# bg = lambda text, color: "\33[48;5;" + str(color) + "m" + text + "\33[0m"

# Simple usage: print(fg("text", 160))
