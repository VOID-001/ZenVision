import React from "react";

const Hero = () => {
  return (
    <div>
      <section class="px-2 py-32 bg-white md:px-0">
        <div class="container items-center max-w-6xl px-8 mx-auto xl:px-5">
          <div class="flex flex-wrap items-center sm:-mx-3">
            <div class="w-full md:w-1/2 md:px-3">
              <div class="w-full pb-6 space-y-6 sm:max-w-md lg:max-w-lg md:space-y-4 lg:space-y-8 xl:space-y-9 sm:pr-5 lg:pr-0 md:pb-0">
                <h1 class="text-4xl font-extrabold tracking-tight text-gray-900 sm:text-5xl md:text-4xl lg:text-5xl xl:text-6xl">
                  <span class="block xl:inline">Personal Guide </span>
                  <span class="block text-indigo-600 xl:inline">
                    Help You Build Healthy Life.
                  </span>
                </h1>
                <p class="mx-auto text-base text-gray-500 sm:max-w-md lg:text-xl md:max-w-3xl">
                  🙏 Yoga helps us connect with our true selves by focusing on
                  mindfulness, breath awareness, and acceptance. 🌞
                </p>
                <div class="relative flex flex-col sm:flex-row sm:space-x-4">
                  <a
                    href="/ai"
                    class="flex items-center w-full px-6 py-3 mb-3 text-lg text-white bg-indigo-600 rounded-md sm:mb-0 hover:bg-indigo-700 sm:w-auto"
                  >
                    Try It Free
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      class="w-5 h-5 ml-1"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    >
                      <line x1="5" y1="12" x2="19" y2="12"></line>
                      <polyline points="12 5 19 12 12 19"></polyline>
                    </svg>
                  </a>
                  <a
                    href="#_"
                    class="flex items-center px-6 py-3 text-gray-500 bg-gray-100 rounded-md hover:bg-gray-200 hover:text-gray-600"
                  >
                    Learn More
                  </a>
                </div>
              </div>
            </div>
            <div class="w-full md:w-1/2">
              <div class="w-full h-auto overflow-hidden rounded-md shadow-xl sm:rounded-xl">
                <img src="https://images.pexels.com/photos/906097/pexels-photo-906097.jpeg" />
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Hero;