import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_show_count = [0]
def _save_show():
    _show_count[0] += 1
    plt.savefig(f"real_{_show_count[0]}.png", dpi=100, bbox_inches='tight')
    plt.close('all')
plt.show = _save_show

exec(open('perceptron_imagen_real.py', encoding='utf-8').read())
print("TEST PASSED!")
