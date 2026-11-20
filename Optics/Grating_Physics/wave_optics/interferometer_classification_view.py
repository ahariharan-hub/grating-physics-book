"""Static interferometer classification schematic for Notebook 7.1.

Presentation only: paths indicate routing, not beam boundaries. No physics
models, interactive state or widget dependencies are introduced.
"""


def classification_schematic_png():
    """Return the wavefront/amplitude division comparison as PNG bytes.

    Notebook 7.1 embeds this static image as a Markdown attachment. To refresh
    it after editing the schematic, replace that attachment's image/png value
    with base64.b64encode(classification_schematic_png()).decode('ascii').
    """
    # Static comparison: arrows and thin paths indicate directions, not beam edges.
    from io import BytesIO
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Arc, Rectangle

    _classification_figure = Figure(figsize=(11, 3.6), dpi=120, layout='constrained')
    FigureCanvasAgg(_classification_figure)
    _classification_axes = _classification_figure.subplots(1, 2)
    for _axis in _classification_axes:
        _axis.set(xlim=(0, 5), ylim=(-1.65, 1.8))
        _axis.set_aspect('equal')
        _axis.axis('off')

    _left, _right = _classification_axes
    _left.set_title('Wavefront division: two openings', fontsize=12)
    for _position in (.25, .65, 1.05):
        _left.plot([_position, _position], [-1.1, 1.1], color='#727c86', lw=1)
    _left.annotate('', xy=(1.4, 0), xytext=(.1, 0),
        arrowprops={'arrowstyle':'->', 'color':'#243444'})
    _left.text(.15, 1.35, 'Incident wave', fontsize=10)
    for _low, _high in ((-1.45, -.7), (-.4, .4), (.7, 1.45)):
        _left.plot([1.65, 1.65], [_low, _high], color='#243444', lw=5)
    for _height, _colour in ((.55, '#2866a3'), (-.55, '#b85c17')):
        for _radius in (.55, 1.15, 1.75, 2.35):
            _arc = Arc((1.65, _height), 2*_radius, 2*_radius,
                theta1=-65, theta2=65, color=_colour, lw=1.2)
            _left.add_patch(_arc)
            _arc.set_clip_path(Rectangle((1.65, -1.1), 2.7, 2.2, transform=_left.transData))
        _left.annotate('', xy=(4.4, 0), xytext=(1.8, _height),
            arrowprops={'arrowstyle':'->', 'linestyle':':', 'color':_colour})
    _left.plot([4.45, 4.45], [-1.1, 1.1], color='#299746', lw=3)
    _left.text(3.5, 1.35, 'Overlap at screen', fontsize=9)
    _left.text(1.1, -1.6, 'Openings → spreading fields', fontsize=10)

    _right.set_title('Amplitude division: split and recombine', fontsize=12)
    _right.annotate('', xy=(1.3, -.55), xytext=(.05, -.55),
        arrowprops={'arrowstyle':'->', 'color':'#243444', 'lw':1.5})
    _right.text(.05, -.25, 'Incident beam', fontsize=9)
    _right.plot([1.3, 3.2, 3.2], [-.55, -.55, .85], color='#2866a3', lw=2)
    _right.plot([1.3, 1.3, 3.2], [-.55, .85, .85], color='#b85c17', lw=2)
    for _xy, _xytext, _colour in (((2.4, -.55), (1.8, -.55), '#2866a3'),
        ((2.4, .85), (1.8, .85), '#b85c17')):
        _right.annotate('', xy=_xy, xytext=_xytext,
            arrowprops={'arrowstyle':'->', 'color':_colour})
    for _cx, _cy in ((1.3, -.55), (3.2, .85)):
        _right.plot([_cx-.2, _cx+.2], [_cy-.2, _cy+.2], color='#79539e', lw=4)
    for _cx, _cy in ((1.3, .85), (3.2, -.55)):
        _right.plot([_cx-.16, _cx+.16], [_cy-.16, _cy+.16], color='#727c86', lw=3)
    _right.annotate('', xy=(4.55, .85), xytext=(3.4, .85),
        arrowprops={'arrowstyle':'->', 'color':'#243444', 'lw':2})
    _right.plot([4.6, 4.6], [.35, 1.35], color='#299746', lw=3)
    _right.text(.75, -1.0, 'Beam splitter', fontsize=9)
    _right.text(2.55, 1.3, 'Recombine → overlap', fontsize=9)
    _right.text(1.7, .05, 'Two optical paths', fontsize=9)
    _right.text(.25, -1.6, 'Generic routing; not a Michelson layout', fontsize=9)
    _classification_figure.suptitle(
        'Schematic paths and arrows are not physical beam boundaries; not to scale', fontsize=10)
    _classification_buffer = BytesIO()
    _classification_figure.savefig(_classification_buffer, format='png', facecolor='white')
    _classification_png = _classification_buffer.getvalue()
    _classification_buffer.close()
    _classification_figure.clear()
    return _classification_png
