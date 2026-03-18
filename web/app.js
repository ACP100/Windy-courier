document.querySelectorAll('video').forEach((video) => {
  video.addEventListener('mouseenter', () => {
    video.play().catch(() => {});
  });
  video.addEventListener('mouseleave', () => {
    video.pause();
  });
});
