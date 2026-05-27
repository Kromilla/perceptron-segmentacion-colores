import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Override plt.show to save instead
_show_count = [0]
_original_show = plt.show
def _save_show():
    _show_count[0] += 1
    plt.savefig(f"output_{_show_count[0]}.png", dpi=100, bbox_inches='tight')
    plt.close('all')
plt.show = _save_show

exec(open('perceptron_colores.py', encoding='utf-8').read())
print("TEST PASSED - All outputs generated successfully!")
