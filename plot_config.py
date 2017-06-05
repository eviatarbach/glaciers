import matplotlib

matplotlib.use('pgf')

import matplotlib.pyplot

params = {'text.usetex': True,
          'pgf.texsystem': 'xelatex',
          'font.family': 'sans-serif',
          'font.sans-serif': ['Optima'],
          'pgf.preamble': [r'\usepackage{mathspec}',
                           (r'\setmathsfont(Digits,Latin,Greek)[Numbers={Lining,Proportional}]'
                            '{Optima}')]}

matplotlib.pyplot.rcParams.update(params)
