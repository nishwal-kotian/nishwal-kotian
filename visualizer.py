import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from config import TradingConfig
from signal_generator import SignalType

class TradingVisualizer:
    """Class to create visualizations for trading algorithm results"""
    
    def __init__(self):
        self.config = TradingConfig()
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_price_and_signals(self, data: pd.DataFrame, symbol: str, 
                              save_path: Optional[str] = None, show_plot: bool = True) -> None:
        """
        Plot price chart with EMA lines and trading signals
        
        Args:
            data: DataFrame with price data and signals
            symbol: Stock symbol
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Main price chart
        ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2, alpha=0.8)
        ax1.plot(data.index, data[f'EMA_{self.config.EMA_SHORT}'], 
                label=f'EMA {self.config.EMA_SHORT}', linewidth=1.5, alpha=0.9)
        ax1.plot(data.index, data[f'EMA_{self.config.EMA_LONG}'], 
                label=f'EMA {self.config.EMA_LONG}', linewidth=1.5, alpha=0.9)
        
        # Mark golden and death crosses
        golden_crosses = data[data['Golden_Cross']]
        death_crosses = data[data['Death_Cross']]
        
        if not golden_crosses.empty:
            ax1.scatter(golden_crosses.index, golden_crosses['Close'], 
                       color='green', marker='^', s=100, label='Golden Cross', zorder=5)
        
        if not death_crosses.empty:
            ax1.scatter(death_crosses.index, death_crosses['Close'], 
                       color='red', marker='v', s=100, label='Death Cross', zorder=5)
        
        # Mark buy and sell signals
        buy_signals = data[data['Signal'] == SignalType.BUY.value]
        sell_signals = data[data['Signal'] == SignalType.SELL.value]
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['Close'], 
                       color='lime', marker='o', s=80, label='Buy Signal', 
                       edgecolors='darkgreen', linewidth=2, zorder=6)
        
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['Close'], 
                       color='orange', marker='o', s=80, label='Sell Signal', 
                       edgecolors='darkred', linewidth=2, zorder=6)
        
        ax1.set_title(f'{symbol} - EMA Crossover Trading Signals', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (INR)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        ax2.bar(data.index, data['Volume'], alpha=0.6, color='skyblue', label='Volume')
        if 'Volume_MA' in data.columns:
            ax2.plot(data.index, data['Volume_MA'], color='red', linewidth=2, 
                    label='Volume MA', alpha=0.8)
        
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_interactive_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create interactive plotly chart
        
        Args:
            data: DataFrame with price data and signals
            symbol: Stock symbol
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxis=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Price and EMAs', 'Volume', 'Signal Strength'),
                           row_heights=[0.6, 0.2, 0.2])
        
        # Price and EMAs
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                mode='lines', name='Close Price',
                                line=dict(color='blue', width=2)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=data.index, y=data[f'EMA_{self.config.EMA_SHORT}'], 
                                mode='lines', name=f'EMA {self.config.EMA_SHORT}',
                                line=dict(color='orange', width=1.5)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=data.index, y=data[f'EMA_{self.config.EMA_LONG}'], 
                                mode='lines', name=f'EMA {self.config.EMA_LONG}',
                                line=dict(color='red', width=1.5)), row=1, col=1)
        
        # Add signal markers
        buy_signals = data[data['Signal'] == SignalType.BUY.value]
        sell_signals = data[data['Signal'] == SignalType.SELL.value]
        
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                                    mode='markers', name='Buy Signal',
                                    marker=dict(symbol='triangle-up', size=10, 
                                              color='green')), row=1, col=1)
        
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                                    mode='markers', name='Sell Signal',
                                    marker=dict(symbol='triangle-down', size=10, 
                                              color='red')), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], 
                            name='Volume', marker_color='lightblue',
                            opacity=0.7), row=2, col=1)
        
        # Signal strength
        if 'Signal_Strength' in data.columns:
            signal_data = data[data['Signal_Strength'] > 0]
            if not signal_data.empty:
                colors = ['green' if s == SignalType.BUY.value else 'red' 
                         for s in signal_data['Signal']]
                fig.add_trace(go.Bar(x=signal_data.index, y=signal_data['Signal_Strength'],
                                    name='Signal Strength', marker_color=colors,
                                    opacity=0.8), row=3, col=1)
        
        fig.update_layout(height=800, title=f'{symbol} - EMA Crossover Analysis',
                         showlegend=True, hovermode='x unified')
        
        return fig
    
    def plot_backtest_results(self, results: Dict, equity_curve: pd.DataFrame, 
                             symbol: str, save_path: Optional[str] = None,
                             show_plot: bool = True) -> None:
        """
        Plot backtest results including equity curve and performance metrics
        
        Args:
            results: Dictionary with backtest results
            equity_curve: DataFrame with portfolio values over time
            symbol: Stock symbol
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[2, 1])
        
        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        if not equity_curve.empty:
            ax1.plot(equity_curve.index, equity_curve['Portfolio_Value'], 
                    linewidth=2, color='blue', label='Portfolio Value')
            ax1.axhline(y=self.config.INITIAL_CAPITAL, color='red', 
                       linestyle='--', alpha=0.7, label='Initial Capital')
            ax1.fill_between(equity_curve.index, equity_curve['Portfolio_Value'], 
                           self.config.INITIAL_CAPITAL, alpha=0.3)
        
        ax1.set_title(f'{symbol} - Portfolio Performance', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (INR)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance metrics text
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('off')
        
        metrics_text = f"""
        PERFORMANCE METRICS
        
        Total Return: ₹{results['total_return']:,.0f} ({results['total_return_pct']:.1f}%)
        Total Trades: {results['total_trades']}
        Win Rate: {results['win_rate']:.1f}%
        Winning Trades: {results['winning_trades']}
        Losing Trades: {results['losing_trades']}
        
        Average Win: ₹{results['avg_win']:,.0f}
        Average Loss: ₹{results['avg_loss']:,.0f}
        Profit Factor: {results['profit_factor']:.2f}
        
        Max Drawdown: {results['max_drawdown']:.1f}%
        Sharpe Ratio: {results['sharpe_ratio']:.2f}
        Calmar Ratio: {results['calmar_ratio']:.2f}
        """
        
        ax2.text(0, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Trade distribution
        ax3 = fig.add_subplot(gs[1, 1])
        if results['total_trades'] > 0:
            trades_df = results['trades_df']
            profits = trades_df['pnl']
            colors = ['green' if p > 0 else 'red' for p in profits]
            ax3.bar(range(len(profits)), profits, color=colors, alpha=0.7)
            ax3.axhline(y=0, color='black', linewidth=1)
            ax3.set_title('Trade P&L Distribution', fontsize=12)
            ax3.set_ylabel('P&L (INR)')
            ax3.set_xlabel('Trade Number')
            ax3.grid(True, alpha=0.3)
        
        # Monthly returns heatmap
        ax4 = fig.add_subplot(gs[2, :])
        if not equity_curve.empty and len(equity_curve) > 30:
            monthly_returns = equity_curve['Portfolio_Value'].resample('M').last().pct_change() * 100
            monthly_returns = monthly_returns.dropna()
            
            if len(monthly_returns) > 1:
                # Create a heatmap-style visualization
                years = monthly_returns.index.year.unique()
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                heatmap_data = np.full((len(years), 12), np.nan)
                
                for i, year in enumerate(years):
                    year_data = monthly_returns[monthly_returns.index.year == year]
                    for month_data in year_data.items():
                        month_idx = month_data[0].month - 1
                        heatmap_data[i, month_idx] = month_data[1]
                
                im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', 
                               vmin=-5, vmax=5)
                
                ax4.set_xticks(range(12))
                ax4.set_xticklabels(months)
                ax4.set_yticks(range(len(years)))
                ax4.set_yticklabels(years)
                ax4.set_title('Monthly Returns Heatmap (%)', fontsize=12)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax4, orientation='horizontal', pad=0.1)
                cbar.set_label('Monthly Return (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_signal_analysis(self, data: pd.DataFrame, symbol: str,
                            save_path: Optional[str] = None, show_plot: bool = True) -> None:
        """
        Plot detailed signal analysis
        
        Args:
            data: DataFrame with signals and indicators
            symbol: Stock symbol
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # EMA difference over time
        ax1 = axes[0, 0]
        if 'EMA_Diff' in data.columns:
            ax1.plot(data.index, data['EMA_Diff'], linewidth=1, alpha=0.8)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax1.fill_between(data.index, data['EMA_Diff'], 0, alpha=0.3)
            
            # Highlight crossover points
            golden_crosses = data[data['Golden_Cross']]
            death_crosses = data[data['Death_Cross']]
            
            if not golden_crosses.empty:
                ax1.scatter(golden_crosses.index, golden_crosses['EMA_Diff'], 
                           color='green', s=50, zorder=5)
            
            if not death_crosses.empty:
                ax1.scatter(death_crosses.index, death_crosses['EMA_Diff'], 
                           color='red', s=50, zorder=5)
        
        ax1.set_title('EMA Difference (Short - Long)')
        ax1.set_ylabel('Price Difference')
        ax1.grid(True, alpha=0.3)
        
        # Signal strength over time
        ax2 = axes[0, 1]
        if 'Signal_Strength' in data.columns:
            signal_data = data[data['Signal_Strength'] > 0]
            if not signal_data.empty:
                colors = ['green' if s == SignalType.BUY.value else 'red' 
                         for s in signal_data['Signal']]
                ax2.scatter(signal_data.index, signal_data['Signal_Strength'], 
                           c=colors, alpha=0.7, s=30)
        
        ax2.set_title('Signal Strength Over Time')
        ax2.set_ylabel('Signal Strength')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Volatility analysis
        ax3 = axes[1, 0]
        if 'Volatility' in data.columns:
            ax3.plot(data.index, data['Volatility'], linewidth=1, alpha=0.8, color='purple')
            # Mark high volatility periods
            high_vol = data['Volatility'].quantile(0.8)
            ax3.axhline(y=high_vol, color='red', linestyle='--', alpha=0.7, 
                       label=f'80th percentile: {high_vol:.2f}')
            ax3.legend()
        
        ax3.set_title('Price Volatility')
        ax3.set_ylabel('Annualized Volatility')
        ax3.grid(True, alpha=0.3)
        
        # Signal frequency histogram
        ax4 = axes[1, 1]
        if 'Signal' in data.columns:
            signal_counts = data['Signal'].value_counts()
            colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
            bars = ax4.bar(signal_counts.index, signal_counts.values, 
                          color=[colors.get(x, 'blue') for x in signal_counts.index])
            
            # Add value labels on bars
            for bar, value in zip(bars, signal_counts.values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                        str(value), ha='center', va='bottom')
        
        ax4.set_title('Signal Frequency')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{symbol} - Signal Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def create_summary_dashboard(self, results_dict: Dict[str, Dict], 
                               save_path: Optional[str] = None) -> None:
        """
        Create a summary dashboard for multiple stocks
        
        Args:
            results_dict: Dictionary with results for multiple stocks
            save_path: Path to save the plot
        """
        if not results_dict:
            print("No results to display")
            return
        
        # Create summary DataFrame
        summary_data = []
        for symbol, results in results_dict.items():
            if results['total_trades'] > 0:
                summary_data.append({
                    'Symbol': symbol.replace('.NS', ''),
                    'Total Return %': results['total_return_pct'],
                    'Total Trades': results['total_trades'],
                    'Win Rate %': results['win_rate'],
                    'Profit Factor': results['profit_factor'],
                    'Max Drawdown %': results['max_drawdown'],
                    'Sharpe Ratio': results['sharpe_ratio']
                })
        
        if not summary_data:
            print("No valid results to display")
            return
        
        summary_df = pd.DataFrame(summary_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Return comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(summary_df['Symbol'], summary_df['Total Return %'], 
                       color=['green' if x > 0 else 'red' for x in summary_df['Total Return %']])
        ax1.set_title('Total Return by Stock', fontweight='bold')
        ax1.set_ylabel('Return (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=1)
        
        # Win rate comparison
        ax2 = axes[0, 1]
        ax2.bar(summary_df['Symbol'], summary_df['Win Rate %'], color='skyblue')
        ax2.set_title('Win Rate by Stock', fontweight='bold')
        ax2.set_ylabel('Win Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50%')
        ax2.legend()
        
        # Profit factor comparison
        ax3 = axes[0, 2]
        ax3.bar(summary_df['Symbol'], summary_df['Profit Factor'], color='lightcoral')
        ax3.set_title('Profit Factor by Stock', fontweight='bold')
        ax3.set_ylabel('Profit Factor')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax3.legend()
        
        # Max drawdown comparison
        ax4 = axes[1, 0]
        ax4.bar(summary_df['Symbol'], summary_df['Max Drawdown %'], color='orange')
        ax4.set_title('Max Drawdown by Stock', fontweight='bold')
        ax4.set_ylabel('Max Drawdown (%)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Sharpe ratio comparison
        ax5 = axes[1, 1]
        ax5.bar(summary_df['Symbol'], summary_df['Sharpe Ratio'], color='mediumpurple')
        ax5.set_title('Sharpe Ratio by Stock', fontweight='bold')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Good (>1)')
        ax5.legend()
        
        # Risk-Return scatter
        ax6 = axes[1, 2]
        scatter = ax6.scatter(summary_df['Max Drawdown %'], summary_df['Total Return %'], 
                             c=summary_df['Sharpe Ratio'], cmap='viridis', s=100, alpha=0.7)
        ax6.set_xlabel('Max Drawdown (%)')
        ax6.set_ylabel('Total Return (%)')
        ax6.set_title('Risk vs Return', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add colorbar for Sharpe ratio
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Sharpe Ratio')
        
        # Add stock labels to scatter plot
        for i, row in summary_df.iterrows():
            ax6.annotate(row['Symbol'], (row['Max Drawdown %'], row['Total Return %']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.suptitle('EMA Crossover Strategy - Performance Summary', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()