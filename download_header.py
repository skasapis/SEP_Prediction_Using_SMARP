import sunpy
import drms
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':

    email = 'xtwang@umich.edu'
    series_type = 'definitive'  # or 'experimental'

    if series_type == 'experimental':
        for tarpnum in tqdm(range(14000)):
            c = drms.Client(debug=True, verbose=True, email=email)
            keys = c.query('su_mbobra.smarp_cea_96m[{:d}][]'.format(tarpnum),
                           key=drms.const.all)
            if(len(keys) > 0):
                keys.to_csv('./header/TARP{:06d}_ATTRS.csv'.format(tarpnum),
                            index=None)
            c, keys = None, None

    if series_type == 'definitive':
        # Here you should get a list of all the SHARP numbers you are working on
        for sharpnum in []:
            c = drms.Client()
            # I checked 'USFLUXL', 'MEANGBL', 'CMASKL'
            # are in the headers already
            keys = c.query('hmi.sharp_cea_720s[{:d}][]'.format(sharpnum),
                           key=drms.const.all)
            print(keys)

            c, keys = None, None

        for tarpnum in [8765]:
            c = drms.Client()
            # For several TARPS I checked, these tree headers are with alot of 0 and NaN
            keys = c.query('mdi.smarp_96m[{:d}][]'.format(tarpnum),
                           key=['USFLUXL', 'MEANGBL', 'CMASKL'])
            print(keys)

            c, keys = None, None
