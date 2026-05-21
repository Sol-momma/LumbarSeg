function setupTabbedShowcase(root) {
  const tabs = [...root.querySelectorAll("[data-showcase-tab]")];
  const panels = [...root.querySelectorAll("[data-showcase-panel]")];

  function activate(id) {
    tabs.forEach((tab) => {
      const selected = tab.dataset.showcaseTab === id;
      tab.classList.toggle("active", selected);
      tab.setAttribute("aria-selected", selected ? "true" : "false");
    });

    panels.forEach((panel) => {
      const selected = panel.dataset.showcasePanel === id;
      panel.classList.toggle("active", selected);
      if (selected) {
        panel.animate(
          [
            { opacity: 0, transform: "translateY(10px)" },
            { opacity: 1, transform: "translateY(0)" },
          ],
          { duration: 260, easing: "cubic-bezier(.2,.8,.2,1)" },
        );
      }
    });
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => activate(tab.dataset.showcaseTab));
  });
}

function setupExpandableVisuals(root = document) {
  root.querySelectorAll("[data-expandable-visual]").forEach((visual) => {
    visual.addEventListener("click", () => {
      visual.classList.toggle("expanded");
    });
  });
}

function setupMetricSwitcher(root) {
  const buttons = [...root.querySelectorAll("[data-metric-tab]")];
  const rows = [...root.querySelectorAll(".metric-row")];

  function activate(metric) {
    buttons.forEach((button) => {
      button.classList.toggle("active", button.dataset.metricTab === metric);
    });

    rows.forEach((row) => {
      const value = metric === "dice" ? row.dataset.paperDice : row.dataset.paperIou;
      row.querySelector("strong").textContent = value || "pending";
      row.animate(
        [
          { opacity: 0.35, transform: "translateX(6px)" },
          { opacity: 1, transform: "translateX(0)" },
        ],
        { duration: 180, easing: "ease-out" },
      );
    });
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => activate(button.dataset.metricTab));
  });
}

function setupScrollDepth() {
  const sections = [...document.querySelectorAll("main > section, .showcase-panel")];
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        entry.target.toggleAttribute("data-in-view", entry.isIntersecting);
      });
    },
    { threshold: 0.24 },
  );

  sections.forEach((section) => observer.observe(section));
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-baseline-showcase]").forEach(setupTabbedShowcase);
  document.querySelectorAll("[data-metric-switcher]").forEach(setupMetricSwitcher);
  setupExpandableVisuals();
  setupScrollDepth();
});
