import React, { forwardRef, useState, useEffect } from 'react';

const LazyLineChart = forwardRef(function LazyLineChart(props, ref) {
  const [LineComponent, setLineComponent] = useState(null);

  useEffect(() => {
    let isMounted = true;
    Promise.all([import('chart.js'), import('react-chartjs-2')])
      .then(([chartJsModule, reactChartJsModule]) => {
        if (!isMounted) return;
        const {
          Chart: ChartJS,
          CategoryScale,
          LinearScale,
          PointElement,
          LineElement,
          Title,
          Tooltip,
          Legend,
          Filler,
        } = chartJsModule;

        ChartJS.register(
          CategoryScale,
          LinearScale,
          PointElement,
          LineElement,
          Title,
          Tooltip,
          Legend,
          Filler
        );

        setLineComponent(() => reactChartJsModule.Line);
      })
      .catch((err) => {
        console.error('Failed to load chart engine dynamic imports:', err);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  if (!LineComponent) {
    return <div className="loading-text">Loading Chart Engine...</div>;
  }

  const Line = LineComponent;
  return <Line ref={ref} {...props} />;
});

export default LazyLineChart;
