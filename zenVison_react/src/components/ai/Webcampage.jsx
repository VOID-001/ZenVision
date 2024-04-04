import Webcam from "react-webcam";
import "./webcampage.css";

const Webcampage = () => {
  return (
    <div className=" w-50 h-50">
      <div className="container">
        <Webcam />
      </div>
    </div>
  );
};

export default Webcampage;
