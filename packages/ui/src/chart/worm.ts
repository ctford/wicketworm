import * as d3 from 'd3';
import type { ProbPoint } from '@wicketworm/shared-types';
import { OVER_OFFSET } from '@wicketworm/shared-types';

export interface InningsBoundary {
  innings: number;
  xOver: number;
  battingTeam: string;
}

export interface WicketFall {
  innings: number;
  xOver: number;
  wickets: number;
  score: string;
}

export interface WormChartOptions {
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  maxOvers?: number;
  inningsBoundaries?: InningsBoundary[];
  wicketFalls?: WicketFall[];
  matchEndOver?: number;  // xOver where match ended
}

export class WormChart {
  private svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private width: number;
  private height: number;
  private margin: { top: number; right: number; bottom: number; left: number };
  private chartWidth: number;
  private chartHeight: number;
  private maxOvers?: number;
  private inningsBoundaries?: InningsBoundary[];
  private wicketFalls?: WicketFall[];
  private matchEndOver?: number;

  constructor(container: string, options: WormChartOptions = {}) {
    this.margin = options.margin ?? { top: 20, right: 120, bottom: 50, left: 50 };
    this.maxOvers = options.maxOvers;
    this.inningsBoundaries = options.inningsBoundaries;
    this.wicketFalls = options.wicketFalls;
    this.matchEndOver = options.matchEndOver;

    const containerEl = document.querySelector(container);
    if (!containerEl) {
      throw new Error(`Container ${container} not found`);
    }

    this.width = options.width ?? containerEl.clientWidth;
    this.height = options.height ?? containerEl.clientHeight;
    this.chartWidth = this.width - this.margin.left - this.margin.right;
    this.chartHeight = this.height - this.margin.top - this.margin.bottom;

    // Clear container
    d3.select(container).selectAll('*').remove();

    // Create SVG
    this.svg = d3.select(container)
      .append('svg')
      .attr('width', this.width)
      .attr('height', this.height)
      .append('g')
      .attr('transform', `translate(${this.margin.left},${this.margin.top})`);
  }

  render(data: ProbPoint[]): void {
    if (data.length === 0) {
      return;
    }

    // Scales - use fixed domain if maxOvers provided
    const xExtent = this.maxOvers
      ? [0, this.maxOvers] as [number, number]
      : d3.extent(data, d => d.xOver) as [number, number];
    const xScale = d3.scaleLinear()
      .domain(xExtent)
      .range([0, this.chartWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([this.chartHeight, 0]);

    // Define areas - stack from bottom (Win) to top (Loss)
    // Use curveStepAfter to show discontinuities at wicket falls (no smoothing)
    const areaWin = d3.area<ProbPoint>()
      .x(d => xScale(d.xOver))
      .y0(this.chartHeight)  // Bottom of chart
      .y1(d => yScale(d.pWin))  // Top of win area
      .curve(d3.curveStepAfter);

    const areaDraw = d3.area<ProbPoint>()
      .x(d => xScale(d.xOver))
      .y0(d => yScale(d.pWin))  // Where win area ended
      .y1(d => yScale(d.pWin + d.pDraw))  // Top of draw area
      .curve(d3.curveStepAfter);

    const areaLoss = d3.area<ProbPoint>()
      .x(d => xScale(d.xOver))
      .y0(d => yScale(d.pWin + d.pDraw))  // Where draw area ended
      .y1(0)  // Top of chart
      .curve(d3.curveStepAfter);

    // Clear previous render
    this.svg.selectAll('*').remove();

    // Render areas first (so lines can be drawn on top)
    this.svg.append('path')
      .datum(data)
      .attr('class', 'area-win')
      .attr('d', areaWin)
      .attr('fill', '#22c55e');

    this.svg.append('path')
      .datum(data)
      .attr('class', 'area-draw')
      .attr('d', areaDraw)
      .attr('fill', '#6b7280');

    this.svg.append('path')
      .datum(data)
      .attr('class', 'area-loss')
      .attr('d', areaLoss)
      .attr('fill', '#ef4444');

    // Wicket fall markers (drawn on top of areas)
    // Line thickness scales with number of wickets that fell
    if (this.wicketFalls) {
      let prevWickets = 0;
      let prevInnings = 0;

      for (const wicket of this.wicketFalls) {
        const xPos = xScale(wicket.xOver);
        if (xPos >= 0 && xPos <= this.chartWidth) {
          // Reset wicket count when innings changes
          if (wicket.innings !== prevInnings) {
            prevWickets = 0;
            prevInnings = wicket.innings;
          }

          // Calculate how many wickets fell in this over
          const wicketsFell = wicket.wickets - prevWickets;
          prevWickets = wicket.wickets;

          // Scale stroke width: 1 wicket = 0.75px, 2 = 1.25px, 3 = 1.75px, etc.
          const strokeWidth = 0.75 + (wicketsFell - 1) * 0.5;

          this.svg.append('line')
            .attr('x1', xPos)
            .attr('x2', xPos)
            .attr('y1', 0)
            .attr('y2', this.chartHeight)
            .attr('stroke', '#9ca3af')
            .attr('stroke-width', strokeWidth)
            .attr('opacity', 0.6);
        }
      }
    }

    // X-axis - show innings markers and every 20 overs
    const maxXOver = d3.max(data, d => d.xOver) ?? 450;
    const overMarks = [];
    for (let i = 0; i <= maxXOver; i += 20) {
      overMarks.push(i);
    }

    const inningsTicks = this.inningsBoundaries?.map(b => b.xOver) ?? [0];
    const allTicks = [...new Set([...overMarks, ...inningsTicks])].sort((a, b) => a - b);

    const xAxis = d3.axisBottom(xScale)
      .tickValues(allTicks)
      .tickFormat(d => {
        const xOver = d.valueOf();
        const boundary = this.inningsBoundaries?.find(b => b.xOver === xOver);
        if (boundary) {
          const team = boundary.battingTeam === 'Australia' ? 'Aus' : 'Eng';
          return `${team} bat`;
        }
        // Show every 20 overs
        if (xOver % 20 === 0) {
          return xOver.toString();
        }
        return '';
      });

    this.svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${this.chartHeight})`)
      .call(xAxis)
      .selectAll('text')
      .style('fill', '#9ca3af')
      .style('font-size', '10px')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end')
      .attr('dx', '-.8em')
      .attr('dy', '.15em');

    this.svg.selectAll('.x-axis path, .x-axis line')
      .style('stroke', '#4b5563');

    // Y-axis
    const yAxis = d3.axisLeft(yScale)
      .tickFormat(d => `${(d.valueOf() * 100).toFixed(0)}%`);

    this.svg.append('g')
      .attr('class', 'y-axis')
      .call(yAxis)
      .selectAll('text')
      .style('fill', '#9ca3af')
      .style('font-size', '12px');

    this.svg.selectAll('.y-axis path, .y-axis line')
      .style('stroke', '#4b5563');

    // Labels
    this.svg.append('text')
      .attr('x', this.chartWidth / 2)
      .attr('y', this.chartHeight + 40)
      .attr('text-anchor', 'middle')
      .style('fill', '#9ca3af')
      .style('font-size', '14px')
      .text('Overs');

    this.svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -this.chartHeight / 2)
      .attr('y', -35)
      .attr('text-anchor', 'middle')
      .style('fill', '#9ca3af')
      .style('font-size', '14px')
      .text('Probability');

    // Legend
    const legend = this.svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${this.chartWidth + 10}, 0)`);

    const legendData = [
      { label: 'Australia', color: '#22c55e' },
      { label: 'Draw', color: '#6b7280' },
      { label: 'England', color: '#ef4444' }
    ];

    legendData.forEach((item, i) => {
      const g = legend.append('g')
        .attr('transform', `translate(0, ${i * 25})`);

      g.append('rect')
        .attr('width', 18)
        .attr('height', 18)
        .attr('fill', item.color);

      g.append('text')
        .attr('x', 24)
        .attr('y', 13)
        .style('fill', '#9ca3af')
        .style('font-size', '14px')
        .text(item.label);
    });

    // Innings boundaries (drawn on top of areas)
    if (this.inningsBoundaries) {
      for (const boundary of this.inningsBoundaries.slice(1)) {  // Skip first boundary at 0
        const xPos = xScale(boundary.xOver);
        if (xPos >= 0 && xPos <= this.chartWidth) {
          this.svg.append('line')
            .attr('x1', xPos)
            .attr('x2', xPos)
            .attr('y1', 0)
            .attr('y2', this.chartHeight)
            .attr('stroke', '#4b5563')
            .attr('stroke-width', 1)
            .attr('stroke-dasharray', '4,4');
        }
      }
    }

    // Match end marker (drawn last as the strongest line)
    if (this.matchEndOver !== undefined) {
      const xPos = xScale(this.matchEndOver);
      if (xPos >= 0 && xPos <= this.chartWidth) {
        this.svg.append('line')
          .attr('x1', xPos)
          .attr('x2', xPos)
          .attr('y1', 0)
          .attr('y2', this.chartHeight)
          .attr('stroke', '#1f2937')  // Dark gray, strongest
          .attr('stroke-width', 2.5)   // Thickest line
          .attr('opacity', 1);
      }
    }

    // Hover interaction - highlight current over and show probabilities
    const highlightGroup = this.svg.append('g').attr('class', 'hover-highlight');
    const tooltipGroup = this.svg.append('g').attr('class', 'hover-tooltip');

    // Add transparent overlay to capture mouse events
    const overlay = this.svg.append('rect')
      .attr('class', 'overlay')
      .attr('width', this.chartWidth)
      .attr('height', this.chartHeight)
      .attr('fill', 'none')
      .attr('pointer-events', 'all');

    // Bisector to find nearest data point
    const bisect = d3.bisector((d: ProbPoint) => d.xOver).left;

    overlay.on('mousemove', (event) => {
      const [mouseX] = d3.pointer(event);
      const xOver = xScale.invert(mouseX);

      // Find the data point whose step we're over (curveStepAfter means use previous point)
      const index = bisect(data, xOver);
      const dataPoint = data[Math.max(0, index - 1)];

      if (!dataPoint) return;

      // Clear previous highlights
      highlightGroup.selectAll('*').remove();
      tooltipGroup.selectAll('*').remove();

      // Calculate step boundaries (from this point to next point)
      let stepStart = xScale(dataPoint.xOver);
      let stepEnd: number;

      // If we're after the match end, highlight entire post-match region as one block
      if (this.matchEndOver !== undefined && xOver >= this.matchEndOver) {
        stepStart = xScale(this.matchEndOver);
        stepEnd = this.chartWidth;
      } else {
        const nextPoint = data[index];
        stepEnd = nextPoint ? xScale(nextPoint.xOver) : this.chartWidth;
      }

      // Draw highlighted areas with brighter colors and borders
      // Win area (bottom)
      highlightGroup.append('rect')
        .attr('x', stepStart)
        .attr('y', yScale(dataPoint.pWin))
        .attr('width', stepEnd - stepStart)
        .attr('height', this.chartHeight - yScale(dataPoint.pWin))
        .attr('fill', '#4ade80')  // Brighter green
        .attr('stroke', '#16a34a')
        .attr('stroke-width', 2);

      // Draw area (middle)
      const drawHeight = yScale(dataPoint.pWin) - yScale(dataPoint.pWin + dataPoint.pDraw);
      highlightGroup.append('rect')
        .attr('x', stepStart)
        .attr('y', yScale(dataPoint.pWin + dataPoint.pDraw))
        .attr('width', stepEnd - stepStart)
        .attr('height', drawHeight)
        .attr('fill', '#9ca3af')  // Brighter gray
        .attr('stroke', '#4b5563')
        .attr('stroke-width', 2);

      // Loss area (top)
      highlightGroup.append('rect')
        .attr('x', stepStart)
        .attr('y', 0)
        .attr('width', stepEnd - stepStart)
        .attr('height', yScale(dataPoint.pWin + dataPoint.pDraw))
        .attr('fill', '#f87171')  // Brighter red
        .attr('stroke', '#dc2626')
        .attr('stroke-width', 2);

      // Show tooltip with probabilities
      const tooltipX = stepStart + (stepEnd - stepStart) / 2;
      const tooltipY = 10;  // Position from top of chart, not above

      const tooltip = tooltipGroup.append('g')
        .attr('transform', `translate(${tooltipX}, ${tooltipY})`);

      // Find wicket info for this block - count all wickets in the highlighted range
      let wicketInfo = '';
      if (this.wicketFalls) {
        // Count wickets that fell in the highlighted region
        const startXOver = xScale.invert(stepStart);
        const endXOver = xScale.invert(stepEnd);

        const wicketsInBlock = this.wicketFalls.filter(w =>
          w.innings === dataPoint.innings &&
          w.xOver > startXOver &&
          w.xOver <= endXOver
        );

        let totalWicketsFell = 0;
        let prevWickets = 0;

        // Find wickets before this block
        const wicketsBefore = this.wicketFalls
          .filter(w => w.innings === dataPoint.innings && w.xOver <= startXOver)
          .sort((a, b) => b.xOver - a.xOver)[0];

        if (wicketsBefore) {
          prevWickets = wicketsBefore.wickets;
        }

        // Count wickets in this block
        if (wicketsInBlock.length > 0) {
          const lastWicketInBlock = wicketsInBlock.sort((a, b) => b.xOver - a.xOver)[0];
          totalWicketsFell = lastWicketInBlock.wickets - prevWickets;
        }

        if (totalWicketsFell > 0) {
          wicketInfo = totalWicketsFell === 1 ? '1 wicket' : `${totalWicketsFell} wickets`;
        }
      }

      // Background rect for tooltip
      // Order from top to bottom of chart to match visualization
      const overText = `#${Math.round(dataPoint.xOver)}`;  // Use xOver for cumulative match overs
      const engText = `ENG ${(dataPoint.pLoss * 100).toFixed(1)}%`;
      const drawText = `Draw ${(dataPoint.pDraw * 100).toFixed(1)}%`;
      const ausText = `AUS ${(dataPoint.pWin * 100).toFixed(1)}%`;

      const tooltipTexts = [overText, engText, drawText, ausText];
      if (wicketInfo) {
        tooltipTexts.push(wicketInfo);
      }

      const padding = 8;
      const lineHeight = 16;
      const maxWidth = Math.max(...tooltipTexts.map(t => t.length * 7)) + padding * 2;
      const height = tooltipTexts.length * lineHeight + padding * 2;

      tooltip.append('rect')
        .attr('x', -maxWidth / 2)
        .attr('y', 0)
        .attr('width', maxWidth)
        .attr('height', height)
        .attr('fill', '#1f2937')
        .attr('stroke', '#4b5563')
        .attr('stroke-width', 1)
        .attr('rx', 4);

      // Add text lines with appropriate colors
      // Order: over, ENG, Draw, AUS, [wickets]
      tooltipTexts.forEach((text, i) => {
        let color = '#e5e7eb';  // Default light gray
        let fontWeight = '600';

        if (i === 0) {
          // Over number - white and bold
          color = '#ffffff';
          fontWeight = '700';
        } else if (i === 1) {
          // England (top) - bright red
          color = '#f87171';
        } else if (i === 2) {
          // Draw (middle) - gray
          color = '#9ca3af';
        } else if (i === 3) {
          // Australia (bottom) - bright green
          color = '#4ade80';
        } else {
          // Wickets - light gray
          color = '#d1d5db';
        }

        tooltip.append('text')
          .attr('x', 0)
          .attr('y', padding + (i + 1) * lineHeight - 2)
          .attr('text-anchor', 'middle')
          .style('fill', color)
          .style('font-size', '12px')
          .style('font-weight', fontWeight)
          .text(text);
      });
    });

    overlay.on('mouseleave', () => {
      highlightGroup.selectAll('*').remove();
      tooltipGroup.selectAll('*').remove();
    });
  }
}
