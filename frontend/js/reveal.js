// Landing-page motion: sections fade and rise as they scroll into view, a thin progress line follows the scroll,
// the top bar gains a shadow once the page moves, and the signal-wave dividers draw themselves.
// Everything is skipped for visitors who ask for reduced motion, and content is visible without it.
(function () {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const els = [...document.querySelectorAll(".reveal")];
  if (reduce || !("IntersectionObserver" in window)) { els.forEach((e) => e.classList.add("in")); }
  else {
    const io = new IntersectionObserver((entries) => entries.forEach((en) => {
      if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
    }), { threshold: 0.14, rootMargin: "0px 0px -6% 0px" });
    els.forEach((e) => io.observe(e));
  }
  const bar = document.getElementById("scrollbar"), top = document.querySelector(".topbar");
  let tick = false;
  function onScroll() {
    if (tick) return; tick = true;
    requestAnimationFrame(() => {
      const h = document.documentElement.scrollHeight - innerHeight;
      if (bar) bar.style.transform = "scaleX(" + (h > 0 ? Math.min(1, scrollY / h) : 0) + ")";
      if (top) top.classList.toggle("lifted", scrollY > 8);
      tick = false;
    });
  }
  addEventListener("scroll", onScroll, { passive: true }); onScroll();
})();
