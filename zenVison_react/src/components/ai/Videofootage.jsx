import React from "react";
// import { video } from "@material-tailwind/react";

const Videofootage = () => {
  return (
    <div>
      <video className="h-90 w-96 rounded-lg video " controls autoPlay muted>
        <source src="src\assets\yoga_pose.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>
    </div>
  );
};

export default Videofootage;
