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
        eg = {
            'model_type': '',
            'chemical_symbols': None,
        }

        nac = {
            'model_type': '',
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
            'model_type': ReadVal('s'),
            'chemical_symbols': ReadVal('sl'),
        }
        legacy_keyfunc = {
            'natom': ReadVal('i'),
            'model_path': ReadVal('s'),
            'gpu': ReadVal('i'),
        }
        seen_legacy = set()

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key in keyfunc:
                keywords[key] = keyfunc[key](val)
                continue

            if key in legacy_keyfunc:
                # Parse for backwards-compatible syntax checking but ignore value in runtime config.
                _ = legacy_keyfunc[key](val)
                if key not in seen_legacy:
                    print(self._legacy_key_warning(key), file=sys.stderr)
                    seen_legacy.add(key)
                continue

            sys.exit(
                '\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &nequip_%s' % (key, self.key_type))

        return keywords

    def _legacy_key_warning(self, key):
        if key == 'natom':
            return (
                '\n  DeprecationWarning\n'
                '  PyRAI2MD: legacy key `natom` in &nequip_%s is ignored; '
                'atom count is auto-detected from runtime structures. '
                'Please remove this key.'
            ) % self.key_type

        return (
            '\n  DeprecationWarning\n'
            '  PyRAI2MD: legacy key `%s` in &nequip_%s is ignored; '
            'use `&nequip modeldir` and `&nequip gpu` instead. '
            'Please remove this key.'
        ) % (key, self.key_type)

    @staticmethod
    def info(eg, nac):
        summary = """

  NequIP-NAC (Energy + Gradient + NAC)

  &hyperparameters            Energy+Gradient      Nonadiabatic         Spin-orbit
----------------------------------------------------------------------------------------------
  Model type:                 NequIP%-13s NequIP%-13s %-13s
  Chemical symbols:           %-20s %-20s %-20s
----------------------------------------------------------------------------------------------
        """ % (
            eg['model_type'],
            nac['model_type'],
            'n/a',
            str(eg['chemical_symbols']) if eg['chemical_symbols'] else 'auto',
            str(nac['chemical_symbols']) if nac['chemical_symbols'] else 'auto',
            'n/a',
        )

        return summary
