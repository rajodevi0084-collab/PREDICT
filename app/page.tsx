import Link from "next/link";

const links = [
  { href: "/upload", label: "Upload Datasets" },
  { href: "/train", label: "Manage Training" },
  { href: "/predict", label: "Generate Predictions" },
  { href: "/charts", label: "View Charts" }
];

export default function HomePage() {
  return (
    <div className="container">
      <section className="card-grid">
        {links.map((link) => (
          <Link key={link.href} href={link.href} className="nav-card">
            <h2>{link.label}</h2>
            <p>Navigate to the {link.label.toLowerCase()} workspace.</p>
          </Link>
        ))}
      </section>
    </div>
  );
}
