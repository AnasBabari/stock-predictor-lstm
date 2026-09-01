import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Navbar from './Navbar';

describe('Navbar', () => {
  it('renders branding', () => {
    render(<Navbar theme="dark" onToggleTheme={() => {}} />);
    expect(screen.getByText('Stock')).toBeInTheDocument();
    expect(screen.getByText('LSTM')).toBeInTheDocument();
  });

  it('theme toggle remains accessible', () => {
    const onToggle = vi.fn();
    render(<Navbar theme="light" onToggleTheme={onToggle} />);
    fireEvent.click(screen.getByRole('button', { name: 'Toggle theme' }));
    expect(onToggle).toHaveBeenCalledTimes(1);
  });
});
