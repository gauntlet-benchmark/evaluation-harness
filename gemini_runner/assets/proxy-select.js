(() => {
  const marker = "__proxy_select_installed__";
  if (window[marker]) return;
  window[marker] = true;

  const makeProxy = (select) => {
    if (!select || select.dataset.proxySelectBound === "1") return;
    select.dataset.proxySelectBound = "1";

    const input = document.createElement("input");
    input.className = "proxy-select-input";
    input.placeholder = select.options[select.selectedIndex]?.text || "Select option";
    input.value = select.options[select.selectedIndex]?.text || "";
    input.setAttribute("data-proxy-for", select.name || select.id || "select");

    input.addEventListener("focus", () => {
      const options = Array.from(select.options).map((opt) => opt.text).join(" | ");
      input.setAttribute("title", options);
    });

    input.addEventListener("change", () => {
      const normalized = input.value.trim().toLowerCase();
      const option = Array.from(select.options).find(
        (opt) => opt.text.trim().toLowerCase() === normalized
      );
      if (!option) return;
      select.value = option.value;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });

    select.style.display = "none";
    select.insertAdjacentElement("afterend", input);
  };

  const bindAll = () => {
    document.querySelectorAll("select").forEach(makeProxy);
  };

  bindAll();
  new MutationObserver(bindAll).observe(document.documentElement, {
    childList: true,
    subtree: true,
  });
})();
