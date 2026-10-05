// Runs in <head>, before first paint: marks the page as script-enabled (enables the loader and scroll effects) and
// applies the saved colour theme so there is no flash of the wrong theme.
(function () {
  document.documentElement.classList.add("js");
  try {
    var t = localStorage.getItem("sanketsetu-theme");
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  } catch (e) { /* storage blocked: keep the default theme */ }
})();
