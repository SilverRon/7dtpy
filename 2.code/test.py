import glob, os
from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack
import matplotlib as mpl
from astropy import units as u
import speclite.filters
import time
import multiprocessing
from itertools import repeat

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"

'edit'

mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 200
plt.rc('font', family='serif')

from helper import makeSpecColors
from helper import convert_flam2fnu
from helper import get_wollaeger
from helper import convert_app2abs
from helper import convert_abs2app
kncbtbl = get_wollaeger()
from helper import get_bandwidth_table
bdwtbl = get_bandwidth_table()
#	speclite
from helper import get_speclite_med
meds = get_speclite_med()
mlam = meds.effective_wavelengths
mbdw = bdwtbl['bandwidth'][bdwtbl['group']=='Med']*u.Angstrom
from helper import get_speclite_sdss
sdss = get_speclite_sdss()
slam = sdss.effective_wavelengths
sbdw = bdwtbl['bandwidth'][bdwtbl['group']=='SDSS']*u.Angstrom
from helper import get_speclite_jc
jc = get_speclite_jc()
jclam = jc.effective_wavelengths
jcbdw = bdwtbl['bandwidth'][bdwtbl['group']=='Johnson Cousin']*u.Angstrom
#
from helper import get_speclite_lsst
from helper import get_lsst_bandwidth
from helper import get_lsst_depth
lsstbdw = get_lsst_bandwidth()
lsst = get_speclite_lsst()
lsstlam = lsst.effective_wavelengths

from helper import get_7dt_depth
from helper import get_7dt_broadband_depth
from helper import get_kmtnet_depth
from helper import get_ztf_depth
from helper import get_decam_depth
# dptbl = get_7dt_depth(exptime=180)
dptbl = get_7dt_broadband_depth(exptime=180)

def calc_snr(m, ul, sigma=5):
	snr = sigma*10**((ul-m)/5)
	return snr

def convert_snr2magerr(snr):
	merr = 2.5*np.log10(1+1/snr)
	return merr

def calc_GaussianFraction(seeing, optfactor=0.6731, path_plot=None):
	import numpy as np
	import matplotlib.pyplot as plt
	from scipy.special import erf

	# seeing, optfactor= 1.5, 0.6731

	mu = 0.0
	# sigma = fwhm_seeing/2.355
	fwhm2sigma = seeing*2.355
	# optfactor = 0.6731
	sigma = fwhm2sigma*optfactor

	x = np.linspace(-8, 8, 1000)
	y = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2 * sigma**2))
	y_cum = 0.5 * (1 + erf((x - mu)/(np.sqrt(2 * sigma**2))))

	indx_aperture = np.where(
		(x>-sigma*optfactor) &
		(x<+sigma*optfactor)
	)
	xaper = x[indx_aperture]
	yaper = y[indx_aperture]

	frac = np.sum(yaper)/np.sum(y) 
	# print(np.sum(y), np.sum(yaper), frac)

	if path_plot != None:
		plt.plot(x, y, alpha=0.5, label=f'PDF of N(0, {sigma:1.3f})', lw=5)
		plt.plot(xaper, yaper, alpha=1.0, label=f'Aperture ({frac*1e2:.1f}%)', lw=5,)
		plt.xlabel('x', fontsize=20)
		plt.ylabel('f(x)', fontsize=20)
		plt.legend(loc='lower center', fontsize=14)
		# plt.show()
		plt.savefig(path_plot, overwrite=True)
	else:
		pass

	return frac

def add_noise(mu, sigma, nseed, n=10, path_plot=None):
	"""
	mu, sigma = 17.5, 0.1
	n = 10
	"""
	from scipy.stats import norm
	import numpy as np
	
	try:
		x = np.arange(mu-sigma*n, mu+sigma*n, sigma*1e-3)
		y = norm(mu, sigma).pdf(x)

		if path_plot != None:
			resultlist = []
			for i in range(10000):
				xobs = np.random.choice(x, p=y/np.sum(y))
				# print(xobs)
				resultlist.append(xobs)
			plt.axvspan(xmin=mu-sigma, xmax=mu+sigma, alpha=0.5, color='tomato',)
			plt.axvline(x=mu, ls='--', alpha=1.0, color='tomato', lw=3)
			plt.plot(x, y, lw=3, alpha=0.75, color='grey')
			plt.hist(resultlist, lw=3, alpha=0.75, color='k', histtype='step', density=True)
			plt.xlabel(r'$\rm m_{obs}$')
			plt.plot(x, y)
			plt.savefig(path_plot, overwrite=True)
		else:
			pass
		#	more complicated choice with the fixed random seed
		np.random.seed(int((nseed+1)+(mu*1e2)))
		return np.random.choice(x, p=y/np.sum(y))
	except:
		# print('Something goes wrong (add_noise function)')
		return None

def extract_param_kn_sim_cube(knsp):
	part = os.path.basename(knsp).split('_')

	if part[1] == 'TP':
		dshape = 'toroidal'
	elif part[1] == 'TS':
		dshape = 'spherical'
	else:
		dshape = ''

	#	Latitude
	if part[5] == 'wind1':
		lat = 'Axial'
	elif part[5] == 'wind2':
		lat = 'Edge'
	else:
		lat = ''

	#	Ejecta mass for low-Ye [solar mass]
	md = float(part[7].replace('md', ''))

	#	Ejecta velocity for low-Ye [N*c]
	vd = float(part[8].replace('vd', ''))

	#	Ejecta mass for high-Ye [solar mass]
	mw = float(part[9].replace('mw', ''))

	#	Ejecta velocity for high-Ye [N*c]
	vw = float(part[10].replace('vw', ''))

	#	Angle
	angle = float(part[11].replace('angle', ''))

	return dshape, lat, md, vd, mw, vw, angle

def run_table_routine(knsp, d, niteration, aperfrac, magoffset, dptbl, lam, bdw):
	"""
	knsp = 'spectrum table name'
	niteration = 'number of iterations e.g. 1000'
	dptbl = 'depth table for each band'
	lam = 'wavelength for each band'
	bdw = 'bandwidth for each band'
	"""
	#	New kilonova file name
	path_result = "../5.result/kn_sim_cube_obs"
	nknsp = f"{path_result}/{os.path.basename(knsp).replace('.ecsv', f'_obs_d{int(d.value)}_iter{niteration}.7dtmed.ecsv')}"
	nknplot = f"{path_result}/{os.path.basename(nknsp).replace('.ecsv', '.png')}"
	if not(os.path.exists(nknsp)) or not(os.path.exists(nknplot)):
		# print(f"[{ii+1}/{len(knsplist)}] {os.path.basename(knsp)}", end='\r')
		intbl = ascii.read(knsp)
		#	Extract parameters
		dshape, lat, md, vd, mw, vw, angle = extract_param_kn_sim_cube(knsp)

		#	Setting
		times = np.unique(intbl['t'])
		filterlist = [filte for filte in intbl.keys() if filte != 't']

		tablelist = []
		for nseed in range(niteration):
			# print(f"[{ii+1}/{len(knsplist)}] {int(1e2*(nseed+1)/niteration)}%", end='\r')
			outbl = Table()
			outbl['seed'] = [nseed]*len(times)
			outbl['t'] = times

			for filte in filterlist:
				depth = dptbl['5sigma_depth'][dptbl['name']==filte].item()*u.ABmag

				outbl[f'magabs_{filte}'] = (intbl[filte]+magoffset)*u.ABmag
				outbl[f'magapp_{filte}'] = convert_abs2app(outbl[f'magabs_{filte}'], d.to(u.pc).value)*u.ABmag
				outbl[f'snr_{filte}'] = calc_snr(outbl[f'magapp_{filte}'], depth.value)*aperfrac
				outbl[f'magerr_{filte}'] = convert_snr2magerr(outbl[f'snr_{filte}'])*u.ABmag
				outbl[f'magobs_{filte}'] = [add_noise(mu=m, sigma=merr, nseed=nseed, n=10, path_plot=None) for m, merr in zip(outbl[f'magapp_{filte}'], outbl[f'magerr_{filte}'])]*u.ABmag
				
				outbl[f'fnu_{filte}'] = outbl[f'magapp_{filte}'].to(u.uJy)
				outbl[f'fnuobs_{filte}'] = outbl[f'magobs_{filte}'].to(u.uJy)
				outbl[f'fnuerr_{filte}'] = outbl[f'fnu_{filte}']/outbl[f'snr_{filte}']

				outbl[f'detection_{filte}'] = [True if m<=depth.value else False for m in outbl[f'magobs_{filte}']]

				outbl[f'magabs_{filte}'].format = '1.3f'
				outbl[f'magapp_{filte}'].format = '1.3f'
				outbl[f'magerr_{filte}'].format = '1.3f'
				outbl[f'magobs_{filte}'].format = '1.3f'
				outbl[f'fnu_{filte}'].format = '1.3f'
				outbl[f'fnuobs_{filte}'].format = '1.3f'
				outbl[f'fnuerr_{filte}'].format = '1.3f'
				outbl[f'snr_{filte}'].format = '1.3f'

			#	Table of results --> tabelist
			tablelist.append(outbl)

		#	Stack tables and Put header
		comtbl = vstack(tablelist)
		comtbl.meta['name'] = os.path.basename(knsp)
		for filte in filterlist: comtbl.meta[f'depth_{filte}'] = dptbl['5sigma_depth'][dptbl['name']==filte].item()*u.ABmag
		comtbl.meta['distance'] = d
		comtbl.meta['dshape'] = dshape
		comtbl.meta['lat'] = lat
		comtbl.meta['md'] = md
		comtbl.meta['vd'] = vd
		comtbl.meta['mw'] = mw
		comtbl.meta['vw'] = vw
		comtbl.meta['angle'] = angle

		comtbl.write(nknsp, format='ascii.ecsv', overwrite=True)

		#	Plot
		seeds = np.arange(0, niteration, 1)
		nseed = len(seeds)
		# colors = makeSpecColors(nseed)

	# 	plt.close('all')
	# 	fig = plt.figure(figsize=(20, 4))
	# 	for jj, tt in enumerate(np.arange(8, 32+8, 8)):
	# 		t = times[tt]
	# 		plt.subplot(1, 4, jj+1)

	# 		for ii, seed in enumerate(seeds):
	# 			magobslist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magobs' in key]
	# 			magerrlist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magerr' in key]
	# 			if ii == 0:
	# 				magapplist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magapp' in key]
	# 				plt.errorbar(lam, magapplist, xerr=bdw/2, yerr=magerrlist, marker='.', ms=10, capsize=3, elinewidth=2, c='red', label='Truth', ls='None', alpha=1.0)
	# 			else:
	# 				pass

	# 			plt.errorbar(lam, magobslist, xerr=bdw/2, yerr=magerrlist, ls='None', alpha=0.05, c='grey',)

	# 		xl, xr = plt.xlim()
	# 		yl, yu = plt.ylim()

	# 		plt.errorbar(0, 0, xerr=0, yerr=0, ls='None', alpha=1.0, c='grey', label='Obs')
	# 		plt.legend(loc='lower center', ncol=4, fontsize=14)
	# 		plt.xlim([xl, xr])
	# 		plt.ylim([yu, yl])
	# 		plt.xlabel(r'Wavelength [$\rm \AA$]', fontsize=14)
	# 		plt.ylabel('App. mag [AB]', fontsize=14)
	# 		plt.title(f't={t:1.3f} days')
	# plt.tight_layout()
	# plt.savefig(nknplot)



def run_table_routine47dtugri(knsp, d, niteration, aperfrac, magoffset, dptbl, lam, bdw):
	"""
	knsp = 'spectrum table name'
	niteration = 'number of iterations e.g. 1000'
	dptbl = 'depth table for each band'
	lam = 'wavelength for each band'
	bdw = 'bandwidth for each band'
	"""
	#	New kilonova file name
	path_result = "../5.result/kn_sim_cube_obs"
	nknsp = f"{path_result}/{os.path.basename(knsp).replace('.ecsv', f'_obs_d{int(d.value)}_iter{niteration}.7dtugri.ecsv')}"
	nknplot = f"{path_result}/{os.path.basename(nknsp).replace('.ecsv', '.png')}"
	if not(os.path.exists(nknsp)) or not(os.path.exists(nknplot)):
		# print(f"[{ii+1}/{len(knsplist)}] {os.path.basename(knsp)}", end='\r')
		intbl = ascii.read(knsp)
		#	Extract parameters
		dshape, lat, md, vd, mw, vw, angle = extract_param_kn_sim_cube(knsp)

		#	Setting
		times = np.unique(intbl['t'])
		filterlist = [filte for filte in intbl.keys() if filte != 't']

		tablelist = []
		for nseed in range(niteration):
			# print(f"[{ii+1}/{len(knsplist)}] {int(1e2*(nseed+1)/niteration)}%", end='\r')
			outbl = Table()
			outbl['seed'] = [nseed]*len(times)
			outbl['t'] = times

			for filte in filterlist:
				depth = dptbl['5sigma_depth'][dptbl['name']==filte].item()*u.ABmag

				outbl[f'magabs_{filte}'] = (intbl[filte]+magoffset)*u.ABmag
				outbl[f'magapp_{filte}'] = convert_abs2app(outbl[f'magabs_{filte}'], d.to(u.pc).value)*u.ABmag
				outbl[f'snr_{filte}'] = calc_snr(outbl[f'magapp_{filte}'], depth.value)*aperfrac
				outbl[f'magerr_{filte}'] = convert_snr2magerr(outbl[f'snr_{filte}'])*u.ABmag
				outbl[f'magobs_{filte}'] = [add_noise(mu=m, sigma=merr, nseed=nseed, n=10, path_plot=None) for m, merr in zip(outbl[f'magapp_{filte}'], outbl[f'magerr_{filte}'])]*u.ABmag
				
				outbl[f'fnu_{filte}'] = outbl[f'magapp_{filte}'].to(u.uJy)
				outbl[f'fnuobs_{filte}'] = outbl[f'magobs_{filte}'].to(u.uJy)
				outbl[f'fnuerr_{filte}'] = outbl[f'fnu_{filte}']/outbl[f'snr_{filte}']

				outbl[f'detection_{filte}'] = [True if m<=depth.value else False for m in outbl[f'magobs_{filte}']]

				outbl[f'magabs_{filte}'].format = '1.3f'
				outbl[f'magapp_{filte}'].format = '1.3f'
				outbl[f'magerr_{filte}'].format = '1.3f'
				outbl[f'magobs_{filte}'].format = '1.3f'
				outbl[f'fnu_{filte}'].format = '1.3f'
				outbl[f'fnuobs_{filte}'].format = '1.3f'
				outbl[f'fnuerr_{filte}'].format = '1.3f'
				outbl[f'snr_{filte}'].format = '1.3f'

			#	Table of results --> tabelist
			tablelist.append(outbl)

		#	Stack tables and Put header
		comtbl = vstack(tablelist)
		comtbl.meta['name'] = os.path.basename(knsp)
		for filte in filterlist: comtbl.meta[f'depth_{filte}'] = dptbl['5sigma_depth'][dptbl['name']==filte].item()*u.ABmag
		comtbl.meta['distance'] = d
		comtbl.meta['dshape'] = dshape
		comtbl.meta['lat'] = lat
		comtbl.meta['md'] = md
		comtbl.meta['vd'] = vd
		comtbl.meta['mw'] = mw
		comtbl.meta['vw'] = vw
		comtbl.meta['angle'] = angle

		comtbl.write(nknsp, format='ascii.ecsv', overwrite=True)

		#	Plot
		seeds = np.arange(0, niteration, 1)
		nseed = len(seeds)
		# colors = makeSpecColors(nseed)

	# 	plt.close('all')
	# 	fig = plt.figure(figsize=(20, 4))
	# 	for jj, tt in enumerate(np.arange(8, 32+8, 8)):
	# 		t = times[tt]
	# 		plt.subplot(1, 4, jj+1)

	# 		for ii, seed in enumerate(seeds):
	# 			magobslist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magobs' in key]
	# 			magerrlist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magerr' in key]
	# 			if ii == 0:
	# 				magapplist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magapp' in key]
	# 				plt.errorbar(lam, magapplist, xerr=bdw/2, yerr=magerrlist, marker='.', ms=10, capsize=3, elinewidth=2, c='red', label='Truth', ls='None', alpha=1.0)
	# 			else:
	# 				pass

	# 			plt.errorbar(lam, magobslist, xerr=bdw/2, yerr=magerrlist, ls='None', alpha=0.05, c='grey',)

	# 		xl, xr = plt.xlim()
	# 		yl, yu = plt.ylim()

	# 		plt.errorbar(0, 0, xerr=0, yerr=0, ls='None', alpha=1.0, c='grey', label='Obs')
	# 		plt.legend(loc='lower center', ncol=4, fontsize=14)
	# 		plt.xlim([xl, xr])
	# 		plt.ylim([yu, yl])
	# 		plt.xlabel(r'Wavelength [$\rm \AA$]', fontsize=14)
	# 		plt.ylabel('App. mag [AB]', fontsize=14)
	# 		plt.title(f't={t:1.3f} days')
	# plt.tight_layout()
	# plt.savefig(nknplot)


def run_table_routine4kmtnet(knsp, d, niteration, aperfrac, magoffset, lam, bdw):
	"""
	knsp = 'spectrum table name'
	niteration = 'number of iterations e.g. 1000'
	dptbl = 'depth table for each band'
	lam = 'wavelength for each band'
	bdw = 'bandwidth for each band'
	"""
	#	New kilonova file name
	path_result = "../5.result/kn_sim_cube_obs"
	nknsp = f"{path_result}/{os.path.basename(knsp).replace('.ecsv', f'_obs_d{int(d.value)}_iter{niteration}.kmtnet.ecsv')}"
	nknplot = f"{path_result}/{os.path.basename(nknsp).replace('.ecsv', '.png')}"
	if not(os.path.exists(nknsp)) or not(os.path.exists(nknplot)):
		# print(f"[{ii+1}/{len(knsplist)}] {os.path.basename(knsp)}", end='\r')
		intbl = ascii.read(knsp)
		#	Extract parameters
		dshape, lat, md, vd, mw, vw, angle = extract_param_kn_sim_cube(knsp)

		#	Setting
		times = np.unique(intbl['t'])
		filterlist = [filte for filte in intbl.keys() if filte != 't']

		tablelist = []
		for nseed in range(niteration):
			# print(f"[{ii+1}/{len(knsplist)}] {int(1e2*(nseed+1)/niteration)}%", end='\r')
			outbl = Table()
			outbl['seed'] = [nseed]*len(times)
			outbl['t'] = times

			for filte in filterlist:
				depth = get_kmtnet_depth(filte, exptime=120)*u.ABmag

				outbl[f'magabs_{filte}'] = (intbl[filte]+magoffset)*u.ABmag
				outbl[f'magapp_{filte}'] = convert_abs2app(outbl[f'magabs_{filte}'], d.to(u.pc).value)*u.ABmag
				outbl[f'snr_{filte}'] = calc_snr(outbl[f'magapp_{filte}'], depth.value)*aperfrac
				outbl[f'magerr_{filte}'] = convert_snr2magerr(outbl[f'snr_{filte}'])*u.ABmag
				outbl[f'magobs_{filte}'] = [add_noise(mu=m, sigma=merr, nseed=nseed, n=10, path_plot=None) for m, merr in zip(outbl[f'magapp_{filte}'], outbl[f'magerr_{filte}'])]*u.ABmag
				
				outbl[f'fnu_{filte}'] = outbl[f'magapp_{filte}'].to(u.uJy)
				outbl[f'fnuobs_{filte}'] = outbl[f'magobs_{filte}'].to(u.uJy)
				outbl[f'fnuerr_{filte}'] = outbl[f'fnu_{filte}']/outbl[f'snr_{filte}']

				outbl[f'detection_{filte}'] = [True if m<=depth.value else False for m in outbl[f'magobs_{filte}']]

				outbl[f'magabs_{filte}'].format = '1.3f'
				outbl[f'magapp_{filte}'].format = '1.3f'
				outbl[f'magerr_{filte}'].format = '1.3f'
				outbl[f'magobs_{filte}'].format = '1.3f'
				outbl[f'fnu_{filte}'].format = '1.3f'
				outbl[f'fnuobs_{filte}'].format = '1.3f'
				outbl[f'fnuerr_{filte}'].format = '1.3f'
				outbl[f'snr_{filte}'].format = '1.3f'

			#	Table of results --> tabelist
			tablelist.append(outbl)

		#	Stack tables and Put header
		comtbl = vstack(tablelist)
		comtbl.meta['name'] = os.path.basename(knsp)
		for filte in filterlist: comtbl.meta[f'depth_{filte}'] = get_kmtnet_depth(filte, exptime=120)*u.ABmag
		comtbl.meta['distance'] = d
		comtbl.meta['dshape'] = dshape
		comtbl.meta['lat'] = lat
		comtbl.meta['md'] = md
		comtbl.meta['vd'] = vd
		comtbl.meta['mw'] = mw
		comtbl.meta['vw'] = vw
		comtbl.meta['angle'] = angle

		comtbl.write(nknsp, format='ascii.ecsv', overwrite=True)

		#	Plot
		seeds = np.arange(0, niteration, 1)
		nseed = len(seeds)
		# colors = makeSpecColors(nseed)

	# 	plt.close('all')
	# 	fig = plt.figure(figsize=(20, 4))
	# 	for jj, tt in enumerate(np.arange(8, 32+8, 8)):
	# 		t = times[tt]
	# 		plt.subplot(1, 4, jj+1)

	# 		for ii, seed in enumerate(seeds):
	# 			magobslist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magobs' in key]
	# 			magerrlist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magerr' in key]
	# 			if ii == 0:
	# 				magapplist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magapp' in key]
	# 				plt.errorbar(lam, magapplist, xerr=bdw/2, yerr=magerrlist, marker='.', ms=10, capsize=3, elinewidth=2, c='red', label='Truth', ls='None', alpha=1.0)
	# 			else:
	# 				pass

	# 			plt.errorbar(lam, magobslist, xerr=bdw/2, yerr=magerrlist, ls='None', alpha=0.05, c='grey',)

	# 		xl, xr = plt.xlim()
	# 		yl, yu = plt.ylim()

	# 		plt.errorbar(0, 0, xerr=0, yerr=0, ls='None', alpha=1.0, c='grey', label='Obs')
	# 		plt.legend(loc='lower center', ncol=4, fontsize=14)
	# 		plt.xlim([xl, xr])
	# 		plt.ylim([yu, yl])
	# 		plt.xlabel(r'Wavelength [$\rm \AA$]', fontsize=14)
	# 		plt.ylabel('App. mag [AB]', fontsize=14)
	# 		plt.title(f't={t:1.3f} days')
	# plt.tight_layout()
	# plt.savefig(nknplot)


def run_table_routine4ztf(knsp, d, niteration, aperfrac, magoffset, lam, bdw):
	"""
	knsp = 'spectrum table name'
	niteration = 'number of iterations e.g. 1000'
	dptbl = 'depth table for each band'
	lam = 'wavelength for each band'
	bdw = 'bandwidth for each band'
	"""
	#	New kilonova file name
	path_result = "../5.result/kn_sim_cube_obs"
	nknsp = f"{path_result}/{os.path.basename(knsp).replace('.ecsv', f'_obs_d{int(d.value)}_iter{niteration}.ztf.ecsv')}"
	nknplot = f"{path_result}/{os.path.basename(nknsp).replace('.ecsv', '.png')}"
	if not(os.path.exists(nknsp)) or not(os.path.exists(nknplot)):
		# print(f"[{ii+1}/{len(knsplist)}] {os.path.basename(knsp)}", end='\r')
		intbl = ascii.read(knsp)
		#	Extract parameters
		dshape, lat, md, vd, mw, vw, angle = extract_param_kn_sim_cube(knsp)

		#	Setting
		times = np.unique(intbl['t'])
		# filterlist = [filte for filte in intbl.keys() if filte != 't']
		filterlist = ['g', 'r', 'i']

		tablelist = []
		for nseed in range(niteration):
			# print(f"[{ii+1}/{len(knsplist)}] {int(1e2*(nseed+1)/niteration)}%", end='\r')
			outbl = Table()
			outbl['seed'] = [nseed]*len(times)
			outbl['t'] = times

			for filte in filterlist:
				depth = get_ztf_depth(filte)*u.ABmag

				outbl[f'magabs_{filte}'] = (intbl[filte]+magoffset)*u.ABmag
				outbl[f'magapp_{filte}'] = convert_abs2app(outbl[f'magabs_{filte}'], d.to(u.pc).value)*u.ABmag
				outbl[f'snr_{filte}'] = calc_snr(outbl[f'magapp_{filte}'], depth.value)*aperfrac
				outbl[f'magerr_{filte}'] = convert_snr2magerr(outbl[f'snr_{filte}'])*u.ABmag
				outbl[f'magobs_{filte}'] = [add_noise(mu=m, sigma=merr, nseed=nseed, n=10, path_plot=None) for m, merr in zip(outbl[f'magapp_{filte}'], outbl[f'magerr_{filte}'])]*u.ABmag
				
				outbl[f'fnu_{filte}'] = outbl[f'magapp_{filte}'].to(u.uJy)
				outbl[f'fnuobs_{filte}'] = outbl[f'magobs_{filte}'].to(u.uJy)
				outbl[f'fnuerr_{filte}'] = outbl[f'fnu_{filte}']/outbl[f'snr_{filte}']

				outbl[f'detection_{filte}'] = [True if m<=depth.value else False for m in outbl[f'magobs_{filte}']]

				outbl[f'magabs_{filte}'].format = '1.3f'
				outbl[f'magapp_{filte}'].format = '1.3f'
				outbl[f'magerr_{filte}'].format = '1.3f'
				outbl[f'magobs_{filte}'].format = '1.3f'
				outbl[f'fnu_{filte}'].format = '1.3f'
				outbl[f'fnuobs_{filte}'].format = '1.3f'
				outbl[f'fnuerr_{filte}'].format = '1.3f'
				outbl[f'snr_{filte}'].format = '1.3f'

			#	Table of results --> tabelist
			tablelist.append(outbl)

		#	Stack tables and Put header
		comtbl = vstack(tablelist)
		comtbl.meta['name'] = os.path.basename(knsp)
		for filte in filterlist: comtbl.meta[f'depth_{filte}'] = get_ztf_depth(filte)*u.ABmag
		comtbl.meta['distance'] = d
		comtbl.meta['dshape'] = dshape
		comtbl.meta['lat'] = lat
		comtbl.meta['md'] = md
		comtbl.meta['vd'] = vd
		comtbl.meta['mw'] = mw
		comtbl.meta['vw'] = vw
		comtbl.meta['angle'] = angle

		comtbl.write(nknsp, format='ascii.ecsv', overwrite=True)

		#	Plot
		seeds = np.arange(0, niteration, 1)
		nseed = len(seeds)
		# colors = makeSpecColors(nseed)

	# 	plt.close('all')
	# 	fig = plt.figure(figsize=(20, 4))
	# 	for jj, tt in enumerate(np.arange(8, 32+8, 8)):
	# 		t = times[tt]
	# 		plt.subplot(1, 4, jj+1)

	# 		for ii, seed in enumerate(seeds):
	# 			magobslist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magobs' in key]
	# 			magerrlist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magerr' in key]
	# 			if ii == 0:
	# 				magapplist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magapp' in key]
	# 				plt.errorbar(lam, magapplist, xerr=bdw/2, yerr=magerrlist, marker='.', ms=10, capsize=3, elinewidth=2, c='red', label='Truth', ls='None', alpha=1.0)
	# 			else:
	# 				pass

	# 			plt.errorbar(lam, magobslist, xerr=bdw/2, yerr=magerrlist, ls='None', alpha=0.05, c='grey',)

	# 		xl, xr = plt.xlim()
	# 		yl, yu = plt.ylim()

	# 		plt.errorbar(0, 0, xerr=0, yerr=0, ls='None', alpha=1.0, c='grey', label='Obs')
	# 		plt.legend(loc='lower center', ncol=4, fontsize=14)
	# 		plt.xlim([xl, xr])
	# 		plt.ylim([yu, yl])
	# 		plt.xlabel(r'Wavelength [$\rm \AA$]', fontsize=14)
	# 		plt.ylabel('App. mag [AB]', fontsize=14)
	# 		plt.title(f't={t:1.3f} days')
	# plt.tight_layout()
	# plt.savefig(nknplot)

def run_table_routine4lsst(knsp, d, niteration, aperfrac, magoffset, lam, bdw):
	"""
	knsp = 'spectrum table name'
	niteration = 'number of iterations e.g. 1000'
	dptbl = 'depth table for each band'
	lam = 'wavelength for each band'
	bdw = 'bandwidth for each band'
	"""
	#	New kilonova file name
	path_result = "../5.result/kn_sim_cube_obs"
	nknsp = f"{path_result}/{os.path.basename(knsp).replace('.ecsv', f'_obs_d{int(d.value)}_iter{niteration}.lsst.ecsv')}"
	nknplot = f"{path_result}/{os.path.basename(nknsp).replace('.ecsv', '.png')}"
	if not(os.path.exists(nknsp)) or not(os.path.exists(nknplot)):
		# print(f"[{ii+1}/{len(knsplist)}] {os.path.basename(knsp)}", end='\r')
		intbl = ascii.read(knsp)
		#	Extract parameters
		dshape, lat, md, vd, mw, vw, angle = extract_param_kn_sim_cube(knsp)

		#	Setting
		times = np.unique(intbl['t'])
		# filterlist = [filte for filte in intbl.keys() if filte != 't']
		filterlist = ['g', 'r', 'i']

		tablelist = []
		for nseed in range(niteration):
			# print(f"[{ii+1}/{len(knsplist)}] {int(1e2*(nseed+1)/niteration)}%", end='\r')
			outbl = Table()
			outbl['seed'] = [nseed]*len(times)
			outbl['t'] = times

			for filte in filterlist:
				depth = get_lsst_depth(filte)*u.ABmag

				outbl[f'magabs_{filte}'] = (intbl[filte]+magoffset)*u.ABmag
				outbl[f'magapp_{filte}'] = convert_abs2app(outbl[f'magabs_{filte}'], d.to(u.pc).value)*u.ABmag
				outbl[f'snr_{filte}'] = calc_snr(outbl[f'magapp_{filte}'], depth.value)*aperfrac
				outbl[f'magerr_{filte}'] = convert_snr2magerr(outbl[f'snr_{filte}'])*u.ABmag
				outbl[f'magobs_{filte}'] = [add_noise(mu=m, sigma=merr, nseed=nseed, n=10, path_plot=None) for m, merr in zip(outbl[f'magapp_{filte}'], outbl[f'magerr_{filte}'])]*u.ABmag
				
				outbl[f'fnu_{filte}'] = outbl[f'magapp_{filte}'].to(u.uJy)
				outbl[f'fnuobs_{filte}'] = outbl[f'magobs_{filte}'].to(u.uJy)
				outbl[f'fnuerr_{filte}'] = outbl[f'fnu_{filte}']/outbl[f'snr_{filte}']

				outbl[f'detection_{filte}'] = [True if m<=depth.value else False for m in outbl[f'magobs_{filte}']]

				outbl[f'magabs_{filte}'].format = '1.3f'
				outbl[f'magapp_{filte}'].format = '1.3f'
				outbl[f'magerr_{filte}'].format = '1.3f'
				outbl[f'magobs_{filte}'].format = '1.3f'
				outbl[f'fnu_{filte}'].format = '1.3f'
				outbl[f'fnuobs_{filte}'].format = '1.3f'
				outbl[f'fnuerr_{filte}'].format = '1.3f'
				outbl[f'snr_{filte}'].format = '1.3f'

			#	Table of results --> tabelist
			tablelist.append(outbl)

		#	Stack tables and Put header
		comtbl = vstack(tablelist)
		comtbl.meta['name'] = os.path.basename(knsp)
		for filte in filterlist: comtbl.meta[f'depth_{filte}'] = get_lsst_depth(filte)*u.ABmag
		comtbl.meta['distance'] = d
		comtbl.meta['dshape'] = dshape
		comtbl.meta['lat'] = lat
		comtbl.meta['md'] = md
		comtbl.meta['vd'] = vd
		comtbl.meta['mw'] = mw
		comtbl.meta['vw'] = vw
		comtbl.meta['angle'] = angle

		comtbl.write(nknsp, format='ascii.ecsv', overwrite=True)

		#	Plot
		seeds = np.arange(0, niteration, 1)
		nseed = len(seeds)
		# colors = makeSpecColors(nseed)

	# 	plt.close('all')
	# 	fig = plt.figure(figsize=(20, 4))
	# 	for jj, tt in enumerate(np.arange(8, 32+8, 8)):
	# 		t = times[tt]
	# 		plt.subplot(1, 4, jj+1)

	# 		for ii, seed in enumerate(seeds):
	# 			magobslist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magobs' in key]
	# 			magerrlist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magerr' in key]
	# 			if ii == 0:
	# 				magapplist = [comtbl[key][(comtbl['t']==t) & (comtbl['seed']==seed)].item() for key in comtbl.keys() if 'magapp' in key]
	# 				plt.errorbar(lam, magapplist, xerr=bdw/2, yerr=magerrlist, marker='.', ms=10, capsize=3, elinewidth=2, c='red', label='Truth', ls='None', alpha=1.0)
	# 			else:
	# 				pass

	# 			plt.errorbar(lam, magobslist, xerr=bdw/2, yerr=magerrlist, ls='None', alpha=0.05, c='grey',)

	# 		xl, xr = plt.xlim()
	# 		yl, yu = plt.ylim()

	# 		plt.errorbar(0, 0, xerr=0, yerr=0, ls='None', alpha=1.0, c='grey', label='Obs')
	# 		plt.legend(loc='lower center', ncol=4, fontsize=14)
	# 		plt.xlim([xl, xr])
	# 		plt.ylim([yu, yl])
	# 		plt.xlabel(r'Wavelength [$\rm \AA$]', fontsize=14)
	# 		plt.ylabel('App. mag [AB]', fontsize=14)
	# 		plt.title(f't={t:1.3f} days')
	# plt.tight_layout()
	# plt.savefig(nknplot)


def run_table_routine4decam(knsp, d, niteration, aperfrac, magoffset, lam, bdw):
	"""
	knsp = 'spectrum table name'
	niteration = 'number of iterations e.g. 1000'
	dptbl = 'depth table for each band'
	lam = 'wavelength for each band'
	bdw = 'bandwidth for each band'
	"""
	#	New kilonova file name
	path_result = "../5.result/kn_sim_cube_obs"
	nknsp = f"{path_result}/{os.path.basename(knsp).replace('.ecsv', f'_obs_d{int(d.value)}_iter{niteration}.decam.ecsv')}"
	nknplot = f"{path_result}/{os.path.basename(nknsp).replace('.ecsv', '.png')}"
	if not(os.path.exists(nknsp)) or not(os.path.exists(nknplot)):
		# print(f"[{ii+1}/{len(knsplist)}] {os.path.basename(knsp)}", end='\r')
		intbl = ascii.read(knsp)
		#	Extract parameters
		dshape, lat, md, vd, mw, vw, angle = extract_param_kn_sim_cube(knsp)

		#	Setting
		times = np.unique(intbl['t'])
		# filterlist = [filte for filte in intbl.keys() if filte != 't']
		filterlist = ['i', 'z']

		tablelist = []
		for nseed in range(niteration):
			# print(f"[{ii+1}/{len(knsplist)}] {int(1e2*(nseed+1)/niteration)}%", end='\r')
			outbl = Table()
			outbl['seed'] = [nseed]*len(times)
			outbl['t'] = times

			for filte in filterlist:
				depth = get_decam_depth(filte)*u.ABmag

				outbl[f'magabs_{filte}'] = (intbl[filte]+magoffset)*u.ABmag
				outbl[f'magapp_{filte}'] = convert_abs2app(outbl[f'magabs_{filte}'], d.to(u.pc).value)*u.ABmag
				outbl[f'snr_{filte}'] = calc_snr(outbl[f'magapp_{filte}'], depth.value)*aperfrac
				outbl[f'magerr_{filte}'] = convert_snr2magerr(outbl[f'snr_{filte}'])*u.ABmag
				outbl[f'magobs_{filte}'] = [add_noise(mu=m, sigma=merr, nseed=nseed, n=10, path_plot=None) for m, merr in zip(outbl[f'magapp_{filte}'], outbl[f'magerr_{filte}'])]*u.ABmag
				
				outbl[f'fnu_{filte}'] = outbl[f'magapp_{filte}'].to(u.uJy)
				outbl[f'fnuobs_{filte}'] = outbl[f'magobs_{filte}'].to(u.uJy)
				outbl[f'fnuerr_{filte}'] = outbl[f'fnu_{filte}']/outbl[f'snr_{filte}']

				outbl[f'detection_{filte}'] = [True if m<=depth.value else False for m in outbl[f'magobs_{filte}']]

				outbl[f'magabs_{filte}'].format = '1.3f'
				outbl[f'magapp_{filte}'].format = '1.3f'
				outbl[f'magerr_{filte}'].format = '1.3f'
				outbl[f'magobs_{filte}'].format = '1.3f'
				outbl[f'fnu_{filte}'].format = '1.3f'
				outbl[f'fnuobs_{filte}'].format = '1.3f'
				outbl[f'fnuerr_{filte}'].format = '1.3f'
				outbl[f'snr_{filte}'].format = '1.3f'

			#	Table of results --> tabelist
			tablelist.append(outbl)

		#	Stack tables and Put header
		comtbl = vstack(tablelist)
		comtbl.meta['name'] = os.path.basename(knsp)
		for filte in filterlist: comtbl.meta[f'depth_{filte}'] = get_decam_depth(filte)*u.ABmag
		comtbl.meta['distance'] = d
		comtbl.meta['dshape'] = dshape
		comtbl.meta['lat'] = lat
		comtbl.meta['md'] = md
		comtbl.meta['vd'] = vd
		comtbl.meta['mw'] = mw
		comtbl.meta['vw'] = vw
		comtbl.meta['angle'] = angle

		comtbl.write(nknsp, format='ascii.ecsv', overwrite=True)