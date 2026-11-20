"""Notebook-aligned lesson catalogue. No GUI imports or launch side effects."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Control:
    label: str
    default: float
    low: float
    high: float
    step: float = .1
    integer: bool = False


CONTROLS = {
    'wavelength': Control('Medium wavelength (um)', 1., .4, 2., .05),
    'period': Control('Grating period (um)', 4., 1., 8.),
    'angle': Control('Incidence polar angle (deg)', 15., 0., 65., 1.),
    'azimuth': Control('Azimuth (deg)', 30., 0., 180., 5.),
    'phase': Control('Relative phase (deg)', 0., -180., 180., 10.),
    'time': Control('Time / optical period', 0., 0., 1., .025),
    'width': Control('Aperture / beam width (um)', 2., .5, 5.),
    'count': Control('Number of slits', 5, 1, 15, 1, True),
    'order': Control('Selected order m', 1, -4, 4, 1, True),
    'distance': Control('Downstream distance s (um)', 8., .5, 40., .5),
    'ratio': Control('Trial spatial wavenumber / k', 1., -1.5, 1.5, .1),
    'backward': Control('Backward-wave fraction', 0., 0., 1., .1),
    'family': Control('Family: 0 plane, 1 standing, 2 spherical, 3 cylindrical, 4 Bessel, 5 multipole, 6 evanescent, 7 Gaussian', 0, 0, 7, 1, True),
    'duty': Control('Clear fraction', .5, .1, .9, .05),
    'thickness': Control('Absorber thickness (um)', .3, .05, 1., .05),
    'loss': Control('Absorber extinction kappa', .8, 0., 2., .1),
    'cutoff': Control('Fourier cutoff M (check convergence)', 12, 4, 32, 2, True),
    'na': Control('Objective NA (air)', .7, .1, 1., .05),
    'amplitude': Control('Field amplitude', 1., 0., 2., .1),
    'cycles': Control('Detector exposure / optical period', 3., .05, 6., .1),
    'balance': Control('Second / first amplitude', 1., 0., 2., .1),
    'coherence': Control('Mutual coherence magnitude', 1., 0., 1., .1),
    'waist': Control('Beam waist (um); wavelength fixed at 1 um', 6., 3., 12., .5),
    'pitch_y': Control('Second lattice period (um)', 5., 1., 8., .1),
    'patches': Control('RS patches per transverse axis', 25, 9, 65, 4, True),
    'component': Control('Selected transverse kx / k', .5, -1.25, 1.25, .05),
    'weyl_order': Control('Weyl quadrature order', 24, 8, 64, 8, True),
    'samples': Control('FFT samples per transverse axis', 256, 128, 512, 64, True),
    'window': Control('FFT window width (um)', 32., 24., 48., 4.),
}


@dataclass(frozen=True)
class Lesson:
    notebook: str
    title: str
    model: str
    guided: tuple
    advanced: tuple
    prompt: str
    observe: str
    theory: str
    presets: tuple = ()


def lesson(book, title, model, guided, advanced, prompt, observe, theory, **presets):
    return Lesson(book, title, model, tuple(guided.split()), tuple(advanced.split()),
                  prompt, observe, theory, tuple(presets.items()))


CHAPTER_TITLES = (
    'Scalar foundations', 'Plane waves and wavevectors', 'A wave meets a grating',
    'Why orders appear', 'Evanescent orders', '3D diffraction and pupils',
    'Field to intensity', 'Interference and overlap', 'Field propagation',
)

# Each subnotebook has its own experiment; Chapter 8 has additional bridge lessons.
LESSONS = (
    (
        lesson('0.1', 'Does this field satisfy Helmholtz?', 'helmholtz', 'ratio wavelength', '',
               'Predict which trial wavenumbers make the residual vanish.',
               'Both +k and -k solve the homogeneous equation; changing the medium wavelength changes k.',
               "U = exp(iqx); U'' + k²U = (k²-q²)U. The equation constrains spatial variation, not just amplitude."),
        lesson('0.2', 'Compare solution families', 'families', 'family wavelength', 'time',
               'Which families have flat, curved or stationary phase fronts?',
               'Compare real field and amplitude. Source singularities are excluded from homogeneous Helmholtz.',
               'Plane, standing, spherical, cylindrical, Bessel, multipole and evanescent fields solve Helmholtz. The Gaussian is a separate paraxial approximation. Bessel transverse ratio is 0.6; multipole degree is 1; evanescent kx/k is 1.25; Gaussian waist/lambda is 2.'),
        lesson('0.3', 'The same boundary value, different fields', 'boundary', 'backward phase', 'wavelength',
               'All curves share U(0). Does that alone select the outgoing solution?',
               'The boundary derivative and travelling-wave content change even though U(0) stays fixed.',
               'U = (1-b)exp(ikz) + b exp(-ikz). A radiation condition and suitable boundary data select a physical boundary-value solution.'),
        lesson('0.4', 'Propagate a known aperture field', 'aperture', 'distance width', 'wavelength samples window',
               'Predict whether a narrower aperture produces a broader propagated pattern.',
               'Compare input, ASM output and Fresnel output. Check sampling and window separately.',
               'Fourier components propagate independently in a homogeneous medium. Prescribing an aperture field is a model assumption, not a material scattering solve.'),
        lesson('0.5', 'Solve a physical grating', 'material', 'thickness loss', 'duty cutoff',
               'Does changing material thickness alter only transmission magnitude?',
               'Inspect complex exit coefficients, reconstructed field and R/T/A.',
               'The normal-incidence scalar/TE model matches U and its normal derivative. Illustrative absorber n = 1.5 + i kappa; air on both sides. Increase M to assess convergence.'),
        lesson('0.6', 'One field, two representations', 'coefficients', 'distance cutoff', 'thickness loss duty',
               'Move the output plane. Which coefficients rotate, and which decay?',
               'Propagating orders retain magnitude; evanescent orders decay. Their coherent sum reconstructs the field.',
               'U(x,zref) = sum C_m(zref) exp(ikx,m x). C_m depends on its reference plane. Efficiency additionally contains longitudinal flux factors.', distance=.5),
        lesson('0.7', 'Ideal mask versus physical absorber', 'mask_compare', 'thickness loss', 'duty cutoff',
               'Will making the absorber thicker necessarily produce a better ideal amplitude mask?',
               'Compare complex fields and coefficients using one incident exit-phase reference.',
               'The ideal mask prescribes a real binary transmission. The physical material can reflect, absorb and alter phase. A common phase reference is essential.'),
    ),
    (
        lesson('1.1', 'Oscillation, wavelength and phase', 'wave', 'wavelength time', 'phase angle azimuth',
               'Advance time by one quarter-period. Which direction does the wave move?',
               'A real-field snapshot moves; the phasor amplitude remains fixed.',
               'u(r,t) = Re{U(r) exp(-i omega t)}; U = exp(ik·r). k = 2 pi/lambda, with lambda measured in the medium.', angle=0., azimuth=0.),
        lesson('1.2', 'Infinite plane wave and finite beam', 'beam', 'width distance', 'wavelength',
               'What changes when the illumination has a finite envelope?',
               'Compare a uniform plane-wave profile with a paraxial Gaussian beam and its angular spectrum.',
               'A finite beam contains a range of plane-wave directions. The Gaussian shown is paraxial; its waist is at s=0.', width=4.),
        lesson('1.3', 'Wavefronts are normal to k', 'wave', 'angle time', 'wavelength phase azimuth',
               'Tilt k. How must the constant-phase fronts rotate?',
               'The arrow is the propagation direction; the phase fronts are perpendicular to it.',
               'A wavefront obeys k·r = constant. Time changes the constant and translates the fronts.', azimuth=0.),
        lesson('1.4', 'Rotate a plane wave in 3D', 'wave3d', 'angle azimuth', 'wavelength time phase',
               'Keep the polar angle fixed and rotate the azimuth. Which components of k change?',
               'Link the 3D direction arrow to the phase on the x-y observation plane.',
               'k/k = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)). This introductory view uses propagation toward +z.'),
    ),
    (
        lesson('2.1', 'From one slit to many', 'array', 'count width', 'period wavelength',
               'Increase the slit count. Do diffraction maxima become broader or narrower?',
               'The single-slit envelope multiplies the finite-array factor; normalized shape is not efficiency.',
               'For prescribed identical openings, the far-field amplitude is a single-opening Fourier integral times a finite phasor sum. No material scattering is inferred.'),
        lesson('2.2', 'Illumination writes a surface phase', 'surface', 'angle wavelength', 'period duty',
               'Tilt illumination. Does the aperture geometry change?',
               'The incident phase slope changes while the binary opening pattern stays fixed.',
               'At the surface, Uout(x) = t(x) exp(ikx,inc x) for this prescribed thin-mask model.'),
        lesson('2.3', 'Which orders can propagate?', 'orders', 'period angle', 'wavelength azimuth order',
               'Decrease the pitch. Follow orders as they leave the propagating disk.',
               'Order locations are determined by symmetry; their amplitudes have not been calculated.',
               'kx,m = kx,inc + 2 pi m/d. Real outgoing directions require kx,m² + ky² <= k².', azimuth=0.),
        lesson('2.4', 'Equal spacing, different coordinate scales', 'coordinates', 'period wavelength', 'angle azimuth order',
               'Compare order spacing in kx and direction cosine as wavelength changes.',
               'Delta kx = 2 pi/d; Delta alpha = lambda/d. Equal alpha spacing does not mean equal angular spacing.',
               'Spatial frequency fx = kx/(2 pi); direction cosine alpha = kx/k. These coordinates describe the same order.'),
    ),
    (
        lesson('3.1', 'Close the phase cycle', 'matching', 'component order', 'period wavelength angle',
               'Move the trial outgoing kx until translation by one pitch preserves relative phase.',
               'The phase mismatch closes only at an integer number of 2 pi cycles.',
               '(kx,out-kx,inc)d = 2 pi m. The trial component may fail this periodicity condition; the selected order satisfies it.'),
        lesson('3.2', 'Follow a selected order', 'matching', 'order angle', 'period wavelength component',
               'Change incidence while keeping m fixed. What preserves phase matching?',
               'The selected outgoing transverse component follows the incident phase slope.',
               'Discrete translation constrains tangential phase. It does not determine excitation strength or ensure a propagating normal component.'),
        lesson('3.3', 'Phase cycles and path difference', 'paths', 'order angle', 'period wavelength',
               'Compare the incident and outgoing path projections over one period.',
               'The selected-order path difference is m wavelengths, including for negative m.',
               'd(alpha_out-alpha_inc) = m lambda. For evanescent orders this transverse phase relation remains valid, but no real outgoing ray exists.'),
    ),
    (
        lesson('4.1', 'Build a decaying order', 'evanescent', 'period distance', 'wavelength angle order time',
               'Reduce pitch until the selected order becomes evanescent.',
               'Transverse oscillation survives while amplitude decays downstream.',
               'With s=-z, q=-kz = sqrt(k²-kx²), taking the positive-real or positive-imaginary branch. U = exp(ikx x + iq s).', period=1., angle=30.),
        lesson('4.2', 'Through grazing: phase, decay and flux', 'evanescent', 'period order', 'wavelength angle distance time',
               'Approach cutoff from both sides. Does decay imply absorption?',
               'Decay length diverges at grazing. A single evanescent harmonic carries zero normal scalar flux in a lossless exterior.',
               'For q=i kappa, amplitude decays as exp(-kappa s), with length 1/kappa. Reactive near fields are distinct from material absorption.', period=1.5, angle=30.),
    ),
    (
        lesson('5.1', 'Conical orders in 3D', 'conical', 'angle azimuth', 'period wavelength order',
               'Rotate incidence away from the x-z plane. Which wavevector component stays common to all orders?',
               'A 1D grating preserves ky; the outgoing directions lie on a common cone about the groove axis.',
               'Grooves run along y, so ky,m = ky,inc. Only kx receives integer reciprocal increments. Downstream is -z.'),
        lesson('5.2', 'One order on three coordinate scales', 'coordinates', 'wavelength azimuth', 'period angle order',
               'Select an order and compare its k, frequency and direction-cosine labels.',
               'All three panels represent identical candidates, including nonpropagating ones.',
               'k = 2 pi f and alpha = lambda fx in a unit-index medium. A coordinate projection is not itself a physical pupil plane.'),
        lesson('5.3', 'Two periods, two order integers', 'lattice', 'period pitch_y', 'wavelength angle azimuth',
               'Change only the second period. Which lattice spacing changes?',
               'The transverse reciprocal lattice shifts with incidence; the propagation disk classifies its candidates.',
               'kparallel,mn = kparallel,inc + m b1 + n b2. Geometry does not imply nonzero excitation. Air exteriors; lambda0=lambda.'),
        lesson('5.4', 'From order directions to pupil collection', 'pupil', 'na angle', 'period wavelength azimuth',
               'Change NA. Which propagating orders are collected?',
               'The unit circle is the collection boundary in sigma coordinates. Source position shifts the order pattern.',
               'sigma = kparallel/(k0 NA). Physical pupil positions require an optical mapping such as a sine-condition objective. This lesson computes angular collection only.'),
    ),
    (
        lesson('6.1', 'What does the detector average?', 'averaging', 'cycles phase', 'amplitude',
               'Can two fields with different starting phases produce the same long-exposure intensity?',
               'Short-exposure mean squares depend on phase; the long-time value approaches A²/2.',
               'For u=A cos(omega t-phi), <u²>=A²/2 over complete periods. Relative phasor intensity is defined here as |U|², twice that mean square.'),
        lesson('6.2', 'A tilted field across a detector', 'detector', 'angle time', 'wavelength amplitude phase',
               'Does a sloping wavefront make bright and dark stripes for a single ideal plane wave?',
               'The instantaneous field oscillates across x, but |U|² is uniform.',
               'For one plane wave, changing phase with x does not change phasor magnitude. Fringes require interference or an amplitude envelope.'),
    ),
    (
        lesson('7.1', 'Phase, balance and mutual coherence', 'interference', 'phase coherence', 'balance period distance',
               'Compare translating fringes with reducing their visibility.',
               'Phase translates fringes; coherence and amplitude imbalance change contrast by different mechanisms.',
               'I = I1 + I2 + 2 sqrt(I1 I2) Re{gamma exp(i Delta phi)}. V = 2 sqrt(I1 I2)|gamma|/(I1+I2).'),
        lesson('7.2', 'Interference of grating orders', 'interference', 'order distance', 'period phase balance coherence',
               'Compare m=1 and m=2 interfering with m=0. What happens to fringe period?',
               'The transverse fringe period is d/|m|; moving the detector changes relative longitudinal phase.',
               'The two prescribed orders share one coherent field unless coherence is deliberately reduced. Their amplitudes are assigned, not solved.', order=1),
        lesson('7.3', 'Finite beams must overlap', 'overlap', 'distance waist', 'phase balance',
               'Move the detector downstream. Where can the two beams still form visible fringes?',
               'The coherent sum differs from the sum of intensities only where the finite beams overlap.',
               'Two prescribed paraxial Gaussian order beams have wavelength 1 um, pitch 4 um and orders 0 and 1. Waist planes are perpendicular to their own axes.'),
        lesson('7.4', 'Directions do not determine coefficients', 'interference', 'balance phase', 'period order coherence distance',
               'Keep directions fixed and vary one complex coefficient.',
               'The angular geometry stays fixed while fringe position and visibility change.',
               'Periodicity selects allowed k values. A physical scattering model or prescribed mask supplies C_m. Efficiency contains flux factors; phase affects coherent recombination.'),
    ),
    (
        lesson('8.1', 'Propagation between two points', 'green', 'distance wavelength', '',
               'How do distance and wavelength change the outgoing point response?',
               'Amplitude falls as 1/R; phase accumulates as kR. This is a response kernel, not a complete boundary reconstruction.',
               'The outgoing 3D Green response is G = exp(ikR)/(4 pi R). Complete Green boundary reconstruction also involves normal derivatives.'),
        lesson('8.2', 'Sum Rayleigh-Sommerfeld patches', 'rs', 'distance width', 'patches phase',
               'Predict the effect of adding more boundary patches before calculating.',
               'Complex patch contributions add as vectors. Refine the quadrature independently of physical parameters.',
               'RS I propagates a prescribed field on a plane using its normal-derivative Green kernel. The square aperture is integrated in both transverse dimensions. Lengths are in medium wavelengths.'),
        lesson('8.2', 'Weyl: rebuild a spherical field', 'weyl', 'weyl_order distance', 'width',
               'Increase angular quadrature order. Does the reconstructed complex field approach the reference?',
               'Compare amplitude and complex error, not only intensity.',
               'The Weyl representation includes propagating and evanescent components. This lesson reuses the notebook full 3D quadrature, sampled on a meridional detector line. Lengths are in wavelengths.'),
        lesson('8.3', 'Compare spatial kernels first', 'kernels', 'distance width', 'wavelength',
               'At which distances can quadratic and far-field phases approximate the full kernel?',
               'All methods use the same source point, observation points and common carrier reference.',
               'Fresnel retains quadratic path variation; Fraunhofer omits source-coordinate curvature. The displayed section is a 3D kernel, not a one-dimensional slit propagator.'),
        lesson('8.3', 'Follow one Fourier component', 'multipliers', 'component distance', 'wavelength',
               'Move kx/k beyond one. What does exact ASM do that Fresnel cannot?',
               'Exact ASM decays evanescent components. The Fresnel approximation continues as a unit-magnitude phase factor.',
               'H_ASM = exp[i s sqrt(k²-kx²-ky²)]; H_F = exp(iks) exp[-i s(kx²+ky²)/(2k)]. No Fraunhofer multiplier is implied.'),
        lesson('8.3', 'Then propagate a full aperture', 'aperture', 'distance width', 'wavelength samples window',
               'Use the kernel lessons to predict when ASM and Fresnel will disagree.',
               'Compare complex fields as well as detector intensity. FFT padding and sampling are independent checks.',
               'A prescribed square aperture is propagated in 2D. FFT boundaries are periodic. The full-window complex-field difference measures this calculation, not universal approximation accuracy.'),
    ),
)


def parameters(spec):
    """Return a fresh complete parameter dictionary for a reproducible lesson."""
    result = {key: value.default for key, value in CONTROLS.items()}
    result.update(spec.presets)
    return result
