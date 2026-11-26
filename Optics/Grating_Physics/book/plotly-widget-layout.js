// Plotly 7.1's anywidget entry point calls perform_render directly, so its
// legacy Lumino resize hook is not registered. Measure after attachment and
// fit with real font metrics; keep the Python figure's data unchanged.
const bookPerformRender = MD.prototype.perform_render;
const bookRemove = MD.prototype.remove;
MD.prototype.perform_render = function () {
  const revision = this.model.get('_widget_layout')?.uirevision;
  if (!['two-order-detector', 'conical-spatial', 'conical-incidence-wavefronts'].includes(revision)) {
    return bookPerformRender.apply(this, arguments);
  }
  this.bookLayoutCleanup?.();
  const node = this.el;
  const container = node.parentElement?.closest('.widget-box') || node.parentElement || node;
  let removed = false;
  let rendered = false;
  let queued = false;
  let busy = false;
  let frame = 0;
  let lastWidth = 0;
  const minimumWidth = Number(this.model.get('_widget_layout')?.width) || 950;
  const targetWidth = () => Math.max(
    minimumWidth,
    container.clientWidth || node.clientWidth,
  );
  const refresh = () => {
    if (removed || !rendered || queued || busy) return;
    queued = true;
    frame = requestAnimationFrame(async () => {
      queued = false;
      if (removed || !node.isConnected || !node.getClientRects().length || !node.clientWidth) return;
      const width = targetWidth();
      if (width === lastWidth) return;
      lastWidth = width;
      busy = true;
      try {
        await node.ownerDocument.fonts.ready;
        if (removed || !node.isConnected) return;
        const layout = {width, _doNotReportToPy: true};
        if (revision === 'conical-spatial') {
          layout['legend.entrywidth'] = 220;
          layout['legend.entrywidthmode'] = 'pixels';
        }
        if (revision === 'conical-incidence-wavefronts') {
          layout['legend.entrywidth'] = 260;
          layout['legend.entrywidthmode'] = 'pixels';
          layout['margin.r'] = 110;
          layout['margin.b'] = 120;
        }
        await Md.default.relayout(node, layout);
      } catch (error) {
        if (!removed && node.isConnected) console.error('Book figure layout failed', error);
      } finally {
        busy = false;
        if (!removed && node.isConnected && node.clientWidth && targetWidth() !== lastWidth) refresh();
      }
    });
  };
  const observer = new ResizeObserver(refresh);
  const afterRender = event => {
    if (event.detail?.element !== node) return;
    rendered = true;
    observer.observe(node);
    if (container !== node) observer.observe(container);
    refresh();
  };
  document.addEventListener('plotlywidget-after-render', afterRender);
  this.bookLayoutCleanup = () => {
    removed = true;
    cancelAnimationFrame(frame);
    observer.disconnect();
    document.removeEventListener('plotlywidget-after-render', afterRender);
  };
  return bookPerformRender.apply(this, arguments);
};
MD.prototype.remove = function () {
  this.bookLayoutCleanup?.();
  return bookRemove.apply(this, arguments);
};
