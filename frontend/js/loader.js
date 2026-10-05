// Hides the boot screen when the page is ready. Never traps the user: a hard cap hides it even if something stalls,
// and after a few seconds it says the connection is slow so a long wait is not mistaken for a hang.
(function () {
  const boot = document.getElementById("boot");
  const root = document.documentElement;
  if (!boot) { root.classList.add("loaded"); return; }
  const started = performance.now(), MIN_MS = 900, MAX_MS = 9000, SLOW_MS = 3500;
  let hidden = false;
  const slowTimer = setTimeout(() => boot.classList.add("slow"), SLOW_MS);
  function hide() {
    if (hidden) return; hidden = true; clearTimeout(slowTimer);
    boot.classList.add("done"); root.classList.add("loaded");
    setTimeout(() => boot.remove(), 800);
  }
  function ready() { setTimeout(hide, Math.max(0, MIN_MS - (performance.now() - started))); }
  if (document.readyState === "complete") ready(); else addEventListener("load", ready, { once: true });
  setTimeout(hide, MAX_MS);
})();
