import { useState } from "react";
import { NextTickPanel } from "../components/NextTickPanel";

const tabs = ["Overview", "Next-Tick"] as const;
type Tab = (typeof tabs)[number];

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<Tab>("Overview");

  return (
    <div>
      <nav>
        {tabs.map((tab) => (
          <button key={tab} onClick={() => setActiveTab(tab)} aria-pressed={activeTab === tab}>
            {tab}
          </button>
        ))}
      </nav>
      <section>
        {activeTab === "Overview" && <p>Select a tab to view details.</p>}
        {activeTab === "Next-Tick" && <NextTickPanel symbol="BTCUSD" />}
      </section>
    </div>
  );
}
