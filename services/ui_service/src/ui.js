(() => {
    // Theme is a browser preference; changing it never makes a backend request.
    const storageKey = "ledger-theme";
    const root = document.documentElement;
    const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");
    let savedTheme = null;
    try { savedTheme = localStorage.getItem(storageKey); } catch (_) { /* Storage may be disabled. */ }

    const applyTheme = (theme) => {
        const dark = theme === "dark";
        root.dataset.ledgerTheme = theme;
        root.style.colorScheme = theme;
        // Keep Gradio's native controls in sync with our custom surfaces.
        [root, document.body, ...document.querySelectorAll("gradio-app, .gradio-container")]
            .forEach((element) => element.classList.toggle("dark", dark));
        const toggle = document.getElementById("theme-toggle");
        if (toggle) toggle.setAttribute("aria-checked", String(dark));
        const label = document.getElementById("theme-label");
        if (label) label.textContent = dark ? "Dark mode" : "Light mode";
    };
    applyTheme(savedTheme === "dark" || savedTheme === "light"
        ? savedTheme : (systemTheme.matches ? "dark" : "light"));

    const closeMobileSidebar = () => {
        if (window.matchMedia("(max-width: 800px)").matches) {
            document.querySelector("#ledger-sidebar.open > .toggle-button")?.click();
        }
    };
    // Gradio injects this script before its components finish mounting.
    const finishMount = () => {
        if (!document.getElementById("theme-toggle") || !document.getElementById("send-btn")) return false;
        applyTheme(root.dataset.ledgerTheme);
        document.getElementById("send-btn").setAttribute("aria-label", "Send question");
        document.getElementById("send-btn").setAttribute("title", "Send question");
        closeMobileSidebar();
        return true;
    };
    const observer = new MutationObserver(() => {
        if (finishMount()) observer.disconnect();
    });
    if (!finishMount()) observer.observe(document.body, { childList: true, subtree: true });

    // Delegation also works if Gradio remounts the header.
    if (window.ledgerThemeCleanup) window.ledgerThemeCleanup();
    const onClick = (event) => {
        if (event.target.closest(".ledger-nav-btn, #new-chat-btn")) closeMobileSidebar();
        if (!event.target.closest("#theme-toggle")) return;
        savedTheme = root.dataset.ledgerTheme === "dark" ? "light" : "dark";
        try { localStorage.setItem(storageKey, savedTheme); } catch (_) { /* Keep the current choice in memory. */ }
        applyTheme(savedTheme);
    };
    const onSystemChange = (event) => {
        if (savedTheme !== "dark" && savedTheme !== "light") applyTheme(event.matches ? "dark" : "light");
    };
    const onStorage = (event) => {
        if (event.key !== storageKey) return;
        savedTheme = event.newValue;
        applyTheme(savedTheme === "dark" || savedTheme === "light"
            ? savedTheme : (systemTheme.matches ? "dark" : "light"));
    };
    document.addEventListener("click", onClick);
    systemTheme.addEventListener("change", onSystemChange);
    window.addEventListener("storage", onStorage);
    window.ledgerThemeCleanup = () => {
        observer.disconnect();
        document.removeEventListener("click", onClick);
        systemTheme.removeEventListener("change", onSystemChange);
        window.removeEventListener("storage", onStorage);
    };
})();
