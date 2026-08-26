const disabled = () => {
  throw new Error('Browser model training is not included in this production build.');
};

export async function trainBrowserForecast() {
  disabled();
}

export async function clearBrowserModelCache() {
  return { cleared: 0, disabled: true };
}
