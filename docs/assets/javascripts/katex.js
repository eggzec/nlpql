/*
 * Render the math that pymdownx.arithmatex emits in generic mode.
 *
 * With `generic: true` the Markdown extension only wraps the formulas
 * in \( ... \) and \[ ... \] and leaves the rendering to the client, so
 * without this hook the raw LaTeX source is what the reader sees.
 *
 * The script works both inside a theme that provides the `document$`
 * observable, where it re-renders after every instant navigation, and
 * in a plain page load.
 */

(function () {
  "use strict";

  var options = {
    delimiters: [
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false }
    ],
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    throwOnError: false
  };

  function render(root) {
    if (typeof window.renderMathInElement !== "function") {
      return false;
    }
    window.renderMathInElement(root || document.body, options);
    return true;
  }

  /* Retry for a short while, the auto-render bundle may still be loading. */
  function renderWhenReady(root) {
    if (render(root)) {
      return;
    }
    var attempts = 0;
    var timer = window.setInterval(function () {
      attempts += 1;
      if (render(root) || attempts > 100) {
        window.clearInterval(timer);
      }
    }, 50);
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(function (doc) {
      renderWhenReady((doc && doc.body) || document.body);
    });
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      renderWhenReady(document.body);
    });
  } else {
    renderWhenReady(document.body);
  }
})();
