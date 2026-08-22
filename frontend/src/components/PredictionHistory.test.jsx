import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import PredictionHistory from './PredictionHistory';

const validEntry = {
  ticker: 'MSFT',
  createdAt: '2026-08-21T12:00:00.000Z',
  horizon: 7,
  forecastType: 'price',
  lastClose: 100,
  predictedValue: 110.5,
  changePercent: 10.5,
  snapshotId: 'abc',
  modelRole: 'browser_learned',
};

describe('PredictionHistory', () => {
  it('renders a valid entry with ticker, prices, horizon and change', () => {
    render(<PredictionHistory items={[validEntry]} onSelectTicker={() => {}} onClearAll={() => {}} />);
    expect(screen.getByText('MSFT')).toBeInTheDocument();
    expect(screen.getByText('$100.00 → $110.50 · 7d')).toBeInTheDocument();
    expect(screen.getByText('▲ +10.50%')).toBeInTheDocument();
    expect(screen.queryByText(/undefined/)).not.toBeInTheDocument();
  });

  it('renders trend entries with the selected direction and horizon', () => {
    const trend = {
      ...validEntry,
      forecastType: 'trend',
      direction: 'Up',
      lastClose: null,
      predictedValue: 0.62,
      changePercent: null,
    };
    render(<PredictionHistory items={[trend]} onSelectTicker={() => {}} onClearAll={() => {}} />);
    expect(screen.getByText('Up · 7d')).toBeInTheDocument();
    expect(screen.getByText('• n/a')).toBeInTheDocument();
  });

  it('renders entries missing optional fields without undefined text', () => {
    const minimal = {
      ticker: 'AAPL',
      createdAt: null,
      horizon: null,
      forecastType: null,
      lastClose: null,
      predictedValue: null,
      changePercent: null,
      snapshotId: null,
      modelRole: null,
    };
    render(<PredictionHistory items={[minimal]} onSelectTicker={() => {}} onClearAll={() => {}} />);
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText('— → — · ?d')).toBeInTheDocument();
    expect(screen.queryByText(/undefined/)).not.toBeInTheDocument();
  });

  it('does not emit duplicate-key warnings for mixed entries', () => {
    const errors = [];
    const errorSpy = vi.spyOn(console, 'error').mockImplementation((...args) => {
      errors.push(args.join(' '));
    });
    const { container } = render(
      <PredictionHistory
        items={[validEntry, { ...validEntry, ticker: 'AAPL', createdAt: null }]}
        onSelectTicker={() => {}}
        onClearAll={() => {}}
      />
    );
    errorSpy.mockRestore();
    expect(container.querySelectorAll('.history-item')).toHaveLength(2);
    expect(errors.join('\n')).not.toMatch(/duplicate key|unique "key"/i);
  });

  it('clicking an entry dispatches the stored ticker', () => {
    const onSelect = vi.fn();
    render(<PredictionHistory items={[validEntry]} onSelectTicker={onSelect} onClearAll={() => {}} />);
    fireEvent.click(screen.getByText('MSFT'));
    expect(onSelect).toHaveBeenCalledWith('MSFT');
  });
});
