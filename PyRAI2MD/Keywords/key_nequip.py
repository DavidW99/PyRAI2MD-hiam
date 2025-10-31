######################################################
#
# PyRAI2MD 2 module for reading input keywords
#
# Author Menghang Wang
# Oct 31 2025
#
######################################################

import sys
from PyRAI2MD.Utils.read_tools import ReadVal

class KeyNequIP:

    def __init__(self, key_type='nac'):
        eg = None

        nac = {
            'model_path': '',
            'gpu': 1,
            'chemical_symbols': None,
        }

        soc = None

        keywords = {
            'eg': eg,
            'nac': nac,
            'soc': soc,
        }

        self.keywords = keywords[key_type]
        self.key_type = key_type

    def default(self):
        return self.keywords

    def update(self, values):
        ## This function read variables from &nequip_eg,&nequip_nac,&nequip_soc
        keywords = self.keywords.copy()
        keyfunc = {
            'model_path': ReadVal('s'),
            'gpu': ReadVal('i'),
            'chemical_symbols': ReadVal('sl'),
        }

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key not in keyfunc.keys():
                sys.exit(
                    '\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &nequip_%s' % (key, self.key_type))
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(nac):
        summary = """

  NequIP-NAC (NAC only)

  &hyperparameters            Energy+Gradient      Nonadiabatic         Spin-orbit
----------------------------------------------------------------------------------------------
  Model path:                 %-20s %-20s %-20s
  GPU:                        %-20s %-20s %-20s
  Chemical symbols:           %-20s %-20s %-20s
----------------------------------------------------------------------------------------------
        """ % (
            'n/a',
            nac['model_path'],
            'n/a',
            'n/a',
            nac['gpu'],
            'n/a',
            'n/a',
            str(nac['chemical_symbols']) if nac['chemical_symbols'] else 'auto',
            'n/a',
        )

        return summary
