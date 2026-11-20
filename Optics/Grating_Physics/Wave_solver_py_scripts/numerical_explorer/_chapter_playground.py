"""Focused Matplotlib chapter experiments: parameters, plots, Apply and Reset."""
import argparse
from pathlib import Path
import textwrap

from _playground import run
from _chapter_examples import LESSONS, CONTROLS, parameters as default_parameters
from _chapter_models import compute, draw

LABELS = {
    'wavelength':'lambda (um)', 'period':'Pitch (um)', 'angle':'Incidence (deg)',
    'azimuth':'Azimuth (deg)', 'phase':'Phase (deg)', 'time':'Time / T',
    'width':'Width (um)', 'count':'Slits', 'order':'Order m', 'distance':'Depth s',
    'ratio':'Trial q / k', 'backward':'Backward fraction', 'family':'Family (0-7)',
    'duty':'Clear fraction', 'thickness':'Thickness (um)', 'loss':'Absorber kappa',
    'cutoff':'Fourier M', 'na':'NA (air)', 'amplitude':'Amplitude', 'cycles':'Exposure / T',
    'balance':'Amplitude ratio', 'coherence':'Coherence |gamma|', 'waist':'Waist (um)',
    'pitch_y':'Pitch y (um)', 'patches':'RS patches / axis', 'component':'kx / k',
    'weyl_order':'Weyl order', 'samples':'FFT samples', 'window':'FFT width (um)',
}


def calculate(chapter, example, settings):
    spec=LESSONS[chapter][example-1]
    params=default_parameters(spec);params.update(settings)
    result=compute(spec,params)
    result['caption']=spec.notebook+' | '+textwrap.fill(spec.prompt,135)
    result['status']=textwrap.fill(result['status'],150)
    return result


def controls_for(spec):
    """Continuous real sliders preserve arbitrary parameter-block reset values."""
    return [(key,LABELS[key],CONTROLS[key].low,CONTROLS[key].high,
             CONTROLS[key].step if CONTROLS[key].integer else None)
            for key in spec.guided+spec.advanced]


def run_chapter(chapter, examples, *, example=1, interactive=True, plot_backend='QtAgg', argv=None):
    parser=argparse.ArgumentParser(description='Edit parameters, run, adjust sliders, Apply.')
    parser.add_argument('--example',type=int,choices=range(1,len(LESSONS[chapter])+1),default=example)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--all',action='store_true',help='Calculate all examples without opening windows')
    parser.add_argument('--render-dir',type=Path,help='Explicitly save static previews')
    args=parser.parse_args(argv)
    if args.headless or args.all or not interactive:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        results=[]
        for index in range(1,len(LESSONS[chapter])+1) if args.all else [args.example]:
            result=calculate(chapter,index,examples[index]);results.append(result)
            print(LESSONS[chapter][index-1].notebook,LESSONS[chapter][index-1].title)
            print(result['status'])
            if args.render_dir:
                args.render_dir.mkdir(parents=True,exist_ok=True)
                figure=Figure(figsize=(12,7),layout='constrained');FigureCanvasAgg(figure)
                draw(result,figure);figure.savefig(args.render_dir/f'{chapter}_{index:02}.png',dpi=110)
        return results
    index=args.example;spec=LESSONS[chapter][index-1]
    params=default_parameters(spec);params.update(examples[index])
    controls=controls_for(spec)
    initial=calculate(chapter,index,params);n=len(initial['panels'])
    print(spec.theory)
    print('Observe:',spec.observe)
    if 'family' in params and spec.model=='families':
        print('Families: 0 plane, 1 standing, 2 spherical, 3 cylindrical, 4 Bessel, 5 multipole, 6 evanescent, 7 Gaussian (paraxial).')
    return run(params,controls,lambda p: calculate(chapter,index,p),draw,
               ((n+1)//2,min(n,2)),spec.notebook+' — '+spec.title,
               interactive=True,plot_backend=plot_backend,plot_bottom=.34,status_y=.245)
