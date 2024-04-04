import React from "react";
import Footer from "./components/landing_page/Footer.jsx";
import Navbar from "./components/landing_page/Navbar.jsx";
import Hero from "./components/landing_page/Hero.jsx";
import Contact from "./components/landing_page/Contact.jsx";
import Slide from "./components/landing_page/Slide.jsx";

const App = () => {
  return (
    <div>
      <Navbar />
      <Hero />
      <Slide />
      <Contact />
      <Footer />
    </div>
  );
};

export default App;
