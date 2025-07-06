// pages/loading.tsx
"use client";
import { motion } from "framer-motion";
import Image from "next/image";
import Head from "next/head";

const LoadingPage = () => {
  return (
    <>
      <Head>
        <title>Loading Animation Test</title>
      </Head>
      <div className="loading-container">
        <div className="animation-wrapper">
          <div className="scene">
            <motion.div
              className="logo-pivot"
              animate={{ scale: [1, 1.05, 1] }}
              transition={{
                duration: 3,
                ease: "easeInOut",
                repeat: Infinity,
              }}
            >
              <Image
                src="/logo.png"
                alt="Loading logo"
                width={80}
                height={80}
                className="logo-image"
              />
            </motion.div>

            <motion.div
              className="orbit"
              animate={{ rotate: 360 }}
              transition={{
                duration: 12,
                ease: "linear",
                repeat: Infinity,
              }}
            >
              <motion.div
                className="spark"
                animate={{ scale: [1, 1.4, 1] }}
                transition={{
                  duration: 2,
                  ease: "easeInOut",
                  repeat: Infinity,
                  repeatDelay: 2,
                }}
              />
            </motion.div>
          </div>
        </div>

        <div className="text-container">
          <span className="loading-text">Finding Your Route</span>
        </div>
      </div>

      <style jsx>{`
        .loading-container {
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          width: 100%;
          height: 100vh;
          background-color: #0a0a10;
          overflow: hidden;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
            Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
        }

        .animation-wrapper {
          position: relative;
          width: 250px;
          height: 250px;
          display: flex;
          justify-content: center;
          align-items: center;
        }

        .scene {
          width: 100%;
          height: 100%;
          position: relative;
          perspective: 1200px;
        }

        .logo-pivot {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          filter: drop-shadow(0 0 15px rgba(0, 191, 255, 0.4));
        }

        .logo-image {
          border-radius: 50%;
        }

        .orbit {
          position: absolute;
          width: 100%;
          height: 100%;
          border-radius: 50%;
          transform-style: preserve-3d;
          transform: rotateX(75deg) rotateY(20deg);
        }

        .spark {
          position: absolute;
          top: -8px;
          left: calc(50% - 8px);
          width: 16px;
          height: 16px;
          background-color: #00bfff;
          border-radius: 50%;
          box-shadow: 0 0 20px 5px #00bfff, 0 0 30px 10px rgba(0, 191, 255, 0.5);
        }

        .spark::after {
          content: "";
          position: absolute;
          top: 50%;
          left: 50%;
          width: 8px;
          height: 150px;
          background: linear-gradient(
            to top,
            transparent,
            rgba(0, 191, 255, 0.5)
          );
          border-radius: 50%;
          transform-origin: top center;
          transform: translate(-50%, -50%) rotate(90deg) translateY(75px);
          opacity: 0.8;
        }

        .text-container {
          margin-top: 40px;
          text-align: center;
        }

        .loading-text {
          font-size: 1.5rem;
          font-weight: 500;
          letter-spacing: 2px;
          background: linear-gradient(
            90deg,
            rgba(255, 255, 255, 0.3),
            #ffffff,
            rgba(255, 255, 255, 0.3)
          );
          background-size: 200% 100%;
          -webkit-background-clip: text;
          background-clip: text;
          color: transparent;
          animation: shimmer 3s linear infinite;
        }

        @keyframes shimmer {
          0% {
            background-position: 200% 0;
          }
          100% {
            background-position: -200% 0;
          }
        }
      `}</style>
    </>
  );
};

export default LoadingPage;
