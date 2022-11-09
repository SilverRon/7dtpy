#	22.09.25
#	Gregory S.H. Paek
#================================================================
#	Library
#----------------------------------------------------------------
from astropy.io import ascii
import numpy as np
import speclite.filters
from astropy import units as u
from astropy.table import Table, vstack, hstack
from astropy import constants as const
import warnings
warnings.filterwarnings('ignore')
#----------------------------------------------------------------
def get_bandwidth_table():
	#	Bandwidth Table
	bdwtbl = Table()
	grouplist = ['Med']*20+['SDSS']*4+['Johnson Cousin']*4
	filterlist = [f"m{int(filte)}" for filte in np.arange(400, 875+25, 25)]+['g', 'i', 'r', 'u']+['B', 'V', 'R', 'I']
	bandwidths = [250]*20+[1370, 1530, 1370, 500]+[781, 991, 1066, 2892]
	bdwtbl['group'] = grouplist
	bdwtbl['filter'] = filterlist
	bdwtbl['bandwidth'] = bandwidths
	return bdwtbl
#----------------------------------------------------------------
def get_lsst_bandwidth():
	"""
	filterlist : ugrizy
	"""
	return np.array([494.43, 1419.37, 1327.32, 1244.00, 1024.11, 930.04])*u.Angstrom
#----------------------------------------------------------------
def makeSpecColors(n, palette='Spectral'):
	#	Color palette
	import seaborn as sns
	palette = sns.color_palette(palette, as_cmap=True,)
	palette.reversed

	clist_ = [palette(i) for i in range(palette.N)]
	cstep = int(len(clist_)/n)
	clist = [clist_[i*cstep] for i in range(n)]
	return clist
#----------------------------------------------------------------
def convert_lam2nu(lam):
	nu = (const.c/(lam)).to(u.Hz)
	return nu
#----------------------------------------------------------------
def convert_fnu2flam(fnu, lam):
	flam = (fnu*const.c/(lam**2)).to((u.erg/((u.cm**2)*u.second*u.Angstrom)))
	return flam
#----------------------------------------------------------------
def convert_flam2fnu(flam, lam):
	fnu = (flam*lam**2/const.c).to((u.erg/((u.cm**2)*u.second*u.Hz)))
	return fnu
#----------------------------------------------------------------
def convert_app2abs(m, d):
	M = m - (5*np.log10(d)-5)
	return M
def convert_abs2app(M, d):
	m = M + (5*np.log10(d)-5)
	return m
#----------------------------------------------------------------
def get_speclite_med():
	rsptbl = ascii.read('../3.table/7dt.filter.response.realistic_optics.ecsv')
	filterlist = np.unique(rsptbl['name'])

	for filte in filterlist:
		#	Filter Table
		fltbl = rsptbl[rsptbl['name']==filte]

		_ = speclite.filters.FilterResponse(
			wavelength = fltbl['lam'],
			response = fltbl['response'], meta=dict(group_name='med', band_name=filte)
		)

	#	New name for speclite class
	mfilterlist = [f"med-{filte}" for filte in filterlist]

	#	Medium filters
	meds = speclite.filters.load_filters(*mfilterlist)
	return meds
#----------------------------------------------------------------
def get_speclite_sdss():
	rsptbl = ascii.read('../3.table/sdss.filter.response.realistic_optics.ecsv')

	filterlist = np.unique(rsptbl['name'])

	# for filte in filterlist:
	for filte in ['u', 'g', 'r', 'i']:
		#	Filter Table
		fltbl = rsptbl[rsptbl['name']==filte]

		_ = speclite.filters.FilterResponse(
			wavelength = fltbl['lam'],
			response = fltbl['response'], meta=dict(group_name='broad', band_name=filte)
		)

	#	New name for speclite class
	bfilterlist = [f"broad-{filte}" for filte in filterlist]

	#	Broad filters
	broads = speclite.filters.load_filters(*bfilterlist)
	return broads
#----------------------------------------------------------------
def get_speclite_jc():
	rsptbl = ascii.read('../3.table/kmtnet/kmtnet_filter.csv')

	# filterlist = np.unique(rsptbl.keys()[1:])
	filterlist = ['B', 'V', 'R', 'I']
	for filte in filterlist:

		rsp = rsptbl[filte]
		rsp = rsp*1e-2 # [%] --> [0.0-1.0]
		rsp[0] = 0.0
		rsp[-1] = 0.0
		rsp[rsp<0] = 0.0

		index = np.where(
			rsp>1e-2	
		)
		
		rsp0 = rsp[index]
		rsp0[0] = 0.0
		rsp0[-1] = 0.0

		#	Filter Table
		_ = speclite.filters.FilterResponse(
			wavelength = rsptbl['wavelength'][index]*u.nm,
			response = rsp0, meta=dict(group_name='kmtnet', band_name=filte)
		)

	#	New name for speclite class
	kfilterlist = [f"kmtnet-{filte}" for filte in filterlist]

	#	KMTNet filters
	kmtns = speclite.filters.load_filters(*kfilterlist)
	return kmtns
#----------------------------------------------------------------
def get_speclite_lsst():
	lsst = speclite.filters.load_filters('lsst2016-*')
	# speclite.filters.plot_filters(lsst, wavelength_limits=(3000, 11000), legend_loc='upper left')
	return lsst
#----------------------------------------------------------------
def get_wollaeger():
	kncbtbl = Table.read(f"../3.table/kn_cube.lite.spectrum.summary.fits")
	return kncbtbl
#----------------------------------------------------------------
def get_7dt_depth(exptime=180):
	dptbl = ascii.read(f"../3.table/7dt.filter.realistic_optics.{exptime}s.summary.ecsv")
	return dptbl
#----------------------------------------------------------------
def get_7dt_broadband_depth(exptime=180):
	dptbl = ascii.read(f"../3.table/sdss.filter.realistic_optics.{exptime}s.summary.ecsv")
	return dptbl
#----------------------------------------------------------------
def get_ztf_depth(filte):
	if filte == 'g':
		depth = 20.8
	elif filte == 'r':
		depth = 20.6
	elif filte == 'i':
		depth = 19.9
	else:
		depth = None
	return depth
#----------------------------------------------------------------
def get_decam_depth(filte):
	if filte == 'i':
		depth = 22.5
	elif filte == 'z':
		depth = 21.8
	else:
		depth = None
	return depth
#----------------------------------------------------------------
def get_lsst_depth(filte):
	if filte == 'u':
		depth = 23.6
	elif filte == 'g':
		depth = 24.7
	elif filte == 'r':
		depth = 24.2
	elif filte == 'i':
		depth = 23.8
	elif filte == 'z':
		depth = 23.2
	elif filte == 'y':
		depth = 22.3
	else:
		depth = None
	return depth
#----------------------------------------------------------------
def get_kmtnet_depth(filte, obs='KMTNet', exptime=120):
	'''
	exptime = 480 [s] : default value for calculating the depth
	'''
	offset = 2.5*np.log10(exptime/480)
	dptbl = Table.read('../3.table/kmtnet/kmtnet.depth.fits')
	obstbl = dptbl[dptbl['obs']==obs]
	try:
		return obstbl['ul5_med'][obstbl['filter']==filte].item()+offset
	except:
		return None
#----------------------------------------------------------------
#	Fitting tools
def func(x, a):
	return a*x

def calc_chisquare(obs, exp):
	return np.sum((obs-exp)**2/exp)
#----------------------------------------------------------------
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