// Mobile menu: the hamburger opens and closes the nav panel. It closes after a link is chosen, on Escape,
// when the page is tapped elsewhere, and when the window grows back to desktop width.
(function () {
  const btn = document.getElementById("menuBtn"), nav = document.getElementById("nav");
  if (!btn || !nav) return;
  function set(open) {
    nav.classList.toggle("open", open);
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    btn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  }
  btn.addEventListener("click", (e) => { e.stopPropagation(); set(!nav.classList.contains("open")); });
  nav.addEventListener("click", (e) => { if (e.target.closest("a")) set(false); });
  document.addEventListener("click", (e) => { if (nav.classList.contains("open") && !e.target.closest(".topbar")) set(false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") set(false); });
  addEventListener("resize", () => { if (innerWidth > 860) set(false); });
  addEventListener("hashchange", () => set(false));
})();

// Scroll-spy and in-page navigation for the landing page.
// The highlighted nav item always follows the section that is under the top bar, whatever the URL says.
// Clicking a nav link scrolls to its section (even when the URL already points there) and keeps the highlight in step.
(function () {
  const IDS = ["home", "problem", "solution", "coverage", "model", "report", "how", "limits", "try"];
  const landing = document.getElementById("view-landing");
  const links = [...document.querySelectorAll("#nav [data-nav]")];
  const OFFSET = 90;            // just below the sticky top bar
  let current = null;
  function paint(id) {
    if (id === current) return; current = id;
    links.forEach((a) => {
      const on = a.dataset.nav === id;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current");
    });
  }
  function spy() {
    if (!landing || landing.classList.contains("hidden")) return;
    let id = "home";
    for (const k of IDS) { const el = document.getElementById(k); if (el && el.getBoundingClientRect().top <= OFFSET) id = k; }
    if (innerHeight + scrollY >= document.documentElement.scrollHeight - 4) id = "limits";   // very bottom: last real section
    paint(id === "try" ? null : id);
  }
  window.spyNav = () => { current = undefined; spy(); };
  let tick = false;
  addEventListener("scroll", () => { if (tick) return; tick = true; requestAnimationFrame(() => { spy(); tick = false; }); }, { passive: true });
  addEventListener("resize", spy);

  document.addEventListener("click", (e) => {
    const a = e.target.closest('a[href^="#/"]'); if (!a || e.defaultPrevented || e.metaKey || e.ctrlKey) return;
    const id = a.getAttribute("href").slice(2);
    if (id === "analyze" || !landing || landing.classList.contains("hidden")) return;   // handled by the router
    const el = id === "" ? null : document.getElementById(id);
    if (id !== "" && !el) return;
    e.preventDefault();
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    window.scrollTo({ top: el ? Math.max(0, el.getBoundingClientRect().top + scrollY - 56) : 0, behavior: reduce ? "auto" : "smooth" });
    try { history.replaceState(null, "", id ? "#/" + id : "#/"); } catch (err) { /* ignore */ }
    paint(id === "" ? "home" : id === "try" ? null : id);
  });
  spy();
})();
