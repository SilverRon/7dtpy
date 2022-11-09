from astropy.io import ascii
import matplotlib.pyplot as plt
fltbl = ascii.read('../3.table/kmtnet/kmtnet_filter.csv')


fig = plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
for filte in ['B', 'V', 'R', 'I']:
	intbl = fltbl[
		(fltbl[filte]>1e-1)
	]
	plt.plot(intbl[intbl.keys()[0]], intbl[filte], lw=3, label=filte, alpha=0.5)
plt.legend(fontsize=20)
plt.title('Johnson-Cousin Filter')
plt.xlim([300, 1200])
plt.ylim([0, 101])

plt.subplot(2, 1, 2)
for filte in ['g', 'r', 'i', 'z']:
	intbl = fltbl[
		(fltbl[filte]>1e-1)
	]
	plt.plot(intbl[intbl.keys()[0]], intbl[filte], lw=3, label=filte, alpha=0.5)
plt.legend(fontsize=20)
plt.xlim([300, 1200])
plt.ylim([0, 101])
plt.title('SDSS Filter')
plt.xlabel('Wavelength [nm]', fontsize=14)

# plt.tight_layout()

plt.savefig('../4.plot/kmtnet.filters.png', dpi=200,)