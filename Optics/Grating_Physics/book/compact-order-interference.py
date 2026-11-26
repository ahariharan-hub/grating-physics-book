"""Presentation for generated lesson 7.2; run after its module reloads."""
if not getattr(DiffractionOrderInterferenceExplorer, '_book_compact', False):
    class _BookOrderInterference(DiffractionOrderInterferenceExplorer):
        _book_compact = True

        def _readout_html(self, vectors, delta, pair, period, visibility, pitch, ratio):
            details = super()._readout_html(vectors, delta, pair, period, visibility, pitch, ratio)
            motion = ('Detector motion: fringes stay fixed.' if abs(delta[2]) < 1e-12 else
                      'Detector motion: fringes shift; spacing stays fixed.')
            return (
                '<div style="font-family:system-ui,sans-serif;font-size:14px;line-height:1.4">'
                '<div style="display:flex;flex-wrap:wrap;gap:8px 28px;align-items:baseline">'
                f'<span><span style="color:{self.order_colours[0]}">●</span> m = {pair[0]:+d}'
                f' &nbsp; <span style="color:{self.order_colours[1]}">●</span> p = {pair[1]:+d}</span>'
                f'<span>Fringe spacing <b>{period:.3f} µm</b></span>'
                f'<span>Visibility <b>{visibility:.3f}</b></span>'
                f'<span>{motion}</span></div>'
                '<details style="margin-top:8px"><summary style="cursor:pointer">'
                'Wavevectors and calculation details</summary>' + details + '</details></div>')

        def __init__(self):
            super().__init__()
            children = self.ui.children
            if len(children) != 10 or children[7] is not self.detector_view.figure:
                raise RuntimeError('Review the changed order-interference widget layout.')
            # Reuse every control and container, preserving ownership and callbacks.
            children[1].children = (self.pitch, self.wavelength, self.angle)
            children[2].children = (self.azimuth, self.pair, self.detector)
            children[4].children = (self.ratio, self.phase, self.reset_button)
            children[6].children = ()
            for index in (0, 3, 5):
                children[index].value = ''
            for index in (0, 3, 5, 6):
                children[index].layout.display = 'none'
            for control in self._controls:
                control.layout.width = '310px'
            for index in (1, 2, 4):
                children[index].layout.gap = '0 8px'
            children[9].value = (
                '<details style="font-size:13px;margin-top:6px"><summary style="cursor:pointer">'
                'How to read the views</summary>' + children[9].value + '</details>')
            self.ui.layout.width = '100%'
            self.ui.layout.max_width = '1000px'
            self.ui.add_class('book-order-interference')
            figure = self.detector_view.figure
            with figure.batch_update():
                figure.update_layout(width=950, height=700, margin=dict(l=35, r=35, t=45, b=65),
                    showlegend=False)
                # Keep the detector square and its physical range tight as the
                # overall figure widens; avoid expanding x to a mostly empty strip.
                figure.update_xaxes(range=[-8, 8], constrain='domain', title_text='')
                figure.update_yaxes(range=[-8, 8], constrain='domain', title_text='')
                self.detector_view.roles['screen'].colorbar.update(
                    x=.76, y=.22, len=.37, thickness=10,
                    title=dict(text=''),
                    tickfont=dict(size=10))
                # Explicit paper positions avoid axis-title measurement in the
                # hidden renderer frame. The concise order key lives in readout.
                labels = []
                for text, x, y, angle, anchor in (
                    ('Detector x (µm)', .5, -.075, 0, 'center'),
                    ('Detector y (µm)', .33, .2021, -90, 'center'),
                    ('Relative intensity', .76, .44, 0, 'left'),
                ):
                    labels.append(dict(text=text, x=x, y=y, xref='paper', yref='paper',
                        xanchor=anchor, yanchor='middle', textangle=angle,
                        showarrow=False, font=dict(size=12)))
                figure.layout.annotations = tuple(figure.layout.annotations) + tuple(labels)

    DiffractionOrderInterferenceExplorer = _BookOrderInterference
