import React from 'react';
// TODO: Import routing provider (e.g., react-router-dom BrowserRouter)
// TODO: Import global styles
// TODO: Import page components

/**
 * App Component
 * Root component that sets up layout and routing for Plant Guard AI.
 */
const App: React.FC = () => {
  return (
    <div className="app-root">
      {/* TODO: Add Navbar component */}
      <header>
        <h1>🌿 Plant Guard AI</h1>
        <p>Intelligent Plant Disease Detection &amp; Severity Assessment</p>
      </header>

      <main>
        {/* TODO: Setup react-router routes here */}
        <section className="hero">
          <h2>Welcome to Plant Guard AI</h2>
          <p>Upload a plant image to detect diseases and get treatment recommendations.</p>
          {/* TODO: Add file upload component */}
        </section>
      </main>

      {/* TODO: Add Footer component */}
      <footer>
        <p>&copy; {new Date().getFullYear()} Plant Guard AI. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default App;
