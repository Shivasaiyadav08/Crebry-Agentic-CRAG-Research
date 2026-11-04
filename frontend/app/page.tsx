import Link from "next/link";
import { Show, SignInButton, SignUpButton, UserButton } from "@clerk/nextjs";
import {
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  FileText,
  Globe2,
  Layers3,
  ScanEye,
  Sparkles,
} from "lucide-react";
import { Brand } from "@/components/brand";
import { ResearchPreview } from "@/components/research-preview";
import { LandingThemeToggle } from "@/components/landing-theme-toggle";
import { ScrollReveal } from "@/components/scroll-reveal";

export default function Page() {
  const authConfigured =
    process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY &&
    process.env.CLERK_SECRET_KEY;
  return (
    <main className="landing">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <nav className="landing-nav" aria-label="Main navigation">
        <Link
          className="landing-logo"
          href="/"
          aria-label="Crebry — The Research Agent home"
        >
          <Brand />
        </Link>
        <div className="landing-nav-links">
          <a href="#workspace">The workspace</a>
          <a href="#how-it-works">How it works</a>
        </div>
        <div className="landing-account">
          <LandingThemeToggle />
          {authConfigured ? (
            <>
              <Show when="signed-out">
                <SignInButton>
                  <button className="landing-auth-button">Sign in</button>
                </SignInButton>
                <SignUpButton>
                  <button className="button button-primary">
                    Get started <ArrowUpRight size={14} />
                  </button>
                </SignUpButton>
              </Show>
              <Show when="signed-in">
                <Link className="button button-primary" href="/research">
                  Your workspace <ArrowUpRight size={14} />
                </Link>
                <UserButton />
              </Show>
            </>
          ) : (
            <>
              <Link className="landing-auth-button" href="/sign-in">
                Sign in
              </Link>
              <Link className="button button-primary" href="/sign-up">
                Get started <ArrowUpRight size={14} />
              </Link>
            </>
          )}
        </div>
      </nav>
      <section className="landing-hero" id="main">
        <div className="hero-copy">
          <div className="hero-kicker">
            <span className="tiny-rule" />
            <span className="kicker-pulse-dot" />
            Adaptive research agent
          </div>
          <h1 className="hero-motion-title">
            Research
            <br />
            <span className="motion-gradient-text">with clarity</span>
            <span className="hero-period">.</span>
          </h1>
          <p className="hero-motion-p">
            Explore questions, synthesize complex sources, and turn information
            into evidence-backed answers — all in one focused workspace.
          </p>
          <div className="landing-actions hero-motion-actions">
            <Link className="landing-cta" href="/research">
              Start researching <ArrowUpRight size={17} />
            </Link>
            <a className="secondary-link" href="#workspace">
              Explore Crebry <ArrowRight size={16} />
            </a>
          </div>
          <div className="hero-footnote hero-motion-footnote">
            <BookOpen size={14} /> Your documents. The wider web. Evidence you
            can inspect.
          </div>
        </div>
        <div className="hero-notebook-wrapper">
          <div className="hero-ambient-glow" aria-hidden="true" />
          <div className="hero-floating-badge badge-top" aria-hidden="true">
            <span className="badge-pulse-dot" />
            <Sparkles size={13} />
            <span>Corrective RAG</span>
          </div>
          <div className="hero-floating-badge badge-bottom" aria-hidden="true">
            <ScanEye size={13} />
            <span>Gemini Vision OCR</span>
          </div>
          <div className="hero-notebook">
            <div className="notebook-top">
              <span>From a question to understanding</span>
              <span>01 / 03</span>
            </div>
            <h2>
              Good questions deserve
              <br />
              more than quick answers.
            </h2>
            <p>
              Bring your sources together. Follow the reasoning. Find what
              matters.
            </p>
            <div className="notebook-source">
              <FileText size={18} />
              <div>
                <strong>Your knowledge</strong>
                <span>Documents, notes, and ideas</span>
              </div>
              <span className="notebook-index">01</span>
            </div>
            <div className="notebook-source">
              <Globe2 size={18} />
              <div>
                <strong>A wider perspective</strong>
                <span>Web research when you need it</span>
              </div>
              <span className="notebook-index">02</span>
            </div>
            <div className="notebook-source">
              <BookOpen size={18} />
              <div>
                <strong>A traceable answer</strong>
                <span>Sources behind the understanding</span>
              </div>
              <span className="notebook-index">03</span>
            </div>
            <div className="notebook-bottom">
              Crebry — The Research Agent{" "}
              <span>
                Built around the evidence <ArrowUpRight size={13} />
              </span>
            </div>
          </div>
        </div>
      </section>
      <ScrollReveal>
        <section
          className="landing-product"
          id="workspace"
          aria-label="Explore the research workspace"
        >
          <div className="product-caption">
            <span>
              <span className="status-dot live" /> A space to think, with
              everything in reach
            </span>
            <span>
              Explore the example below <ArrowRight size={13} />
            </span>
          </div>
          <ResearchPreview />
          <div className="product-bottom">
            <span>One question. Connected sources. A clearer picture.</span>
            <Link href="/research">
              Make it your workspace <ArrowUpRight size={14} />
            </Link>
          </div>
        </section>
      </ScrollReveal>
      <ScrollReveal>
        <section className="landing-method" id="how-it-works">
          <div className="landing-section-title">
            <span className="eyebrow">The research process</span>
            <h2>
              Less searching.
              <br />
              More understanding.
            </h2>
            <p>
              A thoughtful path from the material you have to the answers you
              need.
            </p>
          </div>
          <div className="method-list">
            {[
              {
                n: "01",
                icon: Layers3,
                title: "Give your research a home.",
                text: "Create a workspace for each project. Bring PDFs, documents, and notes together in a library you can return to.",
              },
              {
                n: "02",
                icon: Globe2,
                title: "Ask. Explore. Go deeper.",
                text: "Choose an AI research model and set the minimum document relevance, starting at 50%. Crebry classifies evidence as correct, ambiguous, or incorrect; refines useful passages and corrects gaps with web research when available.",
              },
              {
                n: "03",
                icon: BookOpen,
                title: "See the thinking behind the answer.",
                text: "Read a focused answer with source references. Inspect the original passages, review the research process, and keep asking better questions.",
              },
            ].map((item) => (
              <article key={item.n}>
                <span className="method-number">{item.n}</span>
                <div>
                  <h3>{item.title}</h3>
                  <p>{item.text}</p>
                </div>
                <item.icon size={20} />
              </article>
            ))}
          </div>
        </section>
      </ScrollReveal>
      <ScrollReveal>
        <section className="landing-private" id="your-space">
          <div>
            <span className="eyebrow">Room for your next question</span>
            <h2>
              Your work, connected.
              <br />
              Your thinking, clearer.
            </h2>
          </div>
          <div>
            <p>
              Keep documents, conversations, and discoveries together in your
              personal research workspace. Pick up where you left off.
            </p>
            <Link className="landing-cta" href="/research">
              Start researching <ArrowUpRight size={16} />
            </Link>
            <details className="privacy-details">
              <summary>How Crebry works with your information</summary>
              <p>
                Documents and conversations belong to your account. Document
                embeddings run locally on the server. Relevant passages and
                questions are sent to Gemini; search queries are sent to Tavily.
                Source checks help you inspect answers but cannot guarantee
                every claim is correct.
              </p>
            </details>
          </div>
        </section>
      </ScrollReveal>
      <footer className="landing-footer">
        <Link className="landing-logo" href="/" aria-label="Crebry home">
          <Brand />
        </Link>
        <span>Made for curiosity. Grounded in evidence.</span>
        <a href="#main">
          Back to top <ArrowUpRight size={13} />
        </a>
      </footer>
    </main>
  );
}
