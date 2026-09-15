import { useEffect, useState } from "react";
import { AuthProvider } from "./contexts/AuthContext";
import Home from "./pages/Home";
import Analyze from "./pages/Analyze";
import HowItWorks from "./pages/HowItWorks";
import Technology from "./pages/Technology";
import Trust from "./pages/Trust";
import History from "./pages/History";
import Login from "./pages/Login";
import Signup from "./pages/Signup";

function RouterContent() {
  const [path, setPath] = useState(window.location.pathname);

  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);

    // Global click listener for smooth SPA navigation on internal links
    const handleLinkClick = (e: MouseEvent) => {
      const target = (e.target as HTMLElement).closest("a");
      if (
        target &&
        target.href &&
        target.origin === window.location.origin &&
        !target.hasAttribute("target") &&
        !target.getAttribute("href")?.startsWith("#")
      ) {
        e.preventDefault();
        const url = new URL(target.href);
        if (url.pathname !== window.location.pathname) {
          window.history.pushState({}, "", url.pathname);
          setPath(url.pathname);
          window.scrollTo({ top: 0, behavior: "smooth" });
        }
      }
    };

    document.addEventListener("click", handleLinkClick);
    return () => {
      window.removeEventListener("popstate", onPop);
      document.removeEventListener("click", handleLinkClick);
    };
  }, []);

  if (path === "/analyze") return <Analyze />;
  if (path === "/how-it-works") return <HowItWorks />;
  if (path === "/technology") return <Technology />;
  if (path === "/trust") return <Trust />;
  if (path === "/history") return <History />;
  if (path === "/login") return <Login />;
  if (path === "/signup") return <Signup />;
  return <Home />;
}

export default function App() {
  return (
    <AuthProvider>
      <RouterContent />
    </AuthProvider>
  );
}
