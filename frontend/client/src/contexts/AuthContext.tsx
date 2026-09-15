import { createContext, useContext, useEffect, useState } from "react";

export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  createdAt: string;
}

interface AuthContextType {
  user: User | null;
  login: (email: string, pass: string) => Promise<boolean>;
  signup: (name: string, email: string, pass: string) => Promise<boolean>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    try {
      const stored = localStorage.getItem("signalscope_user");
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });

  const login = async (email: string): Promise<boolean> => {
    // Mock authentication
    const nameFromEmail = email.split("@")[0] || "User";
    const formattedName =
      nameFromEmail.charAt(0).toUpperCase() + nameFromEmail.slice(1);

    const newUser: User = {
      id: "usr_" + Math.random().toString(36).substring(2, 9),
      name: formattedName,
      email,
      createdAt: new Date().toISOString(),
    };

    setUser(newUser);
    localStorage.setItem("signalscope_user", JSON.stringify(newUser));
    return true;
  };

  const signup = async (name: string, email: string): Promise<boolean> => {
    const newUser: User = {
      id: "usr_" + Math.random().toString(36).substring(2, 9),
      name: name || "User",
      email,
      createdAt: new Date().toISOString(),
    };

    setUser(newUser);
    localStorage.setItem("signalscope_user", JSON.stringify(newUser));
    return true;
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem("signalscope_user");
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        login,
        signup,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
