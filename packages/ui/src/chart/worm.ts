import * as d3 from 'd3';
import type { ProbPoint } from '@wicketworm/shared-types';
import { OVER_OFFSET } from '@wicketworm/shared-types';

export interface WormChartOptions {
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  maxOvers?: number;
}

export class WormChart {
  private svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private width: number;
  private height: number;
  private margin: { top: number; right: number; bottom: number; left: number };
  private chartWidth: number;
  private chartHeight: number;
  private maxOvers?: number;

  constructor(container: string, options: WormChartOptions = {}) {
    this.margin = options.margin ?? { top: 20, right: 120, bottom: 50, left: 50 };
    this.maxOvers = options.maxOvers;

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

    // Define areas
    const areaWin = d3.area<ProbPoint>()
      .x(d => xScale(d.xOver))
      .y0(this.chartHeight)
      .y1(d => yScale(1 - d.pWin))
      .curve(d3.curveMonotoneX);

    const areaDraw = d3.area<ProbPoint>()
      .x(d => xScale(d.xOver))
      .y0(d => yScale(1 - d.pWin))
      .y1(d => yScale(1 - d.pWin - d.pDraw))
      .curve(d3.curveMonotoneX);

    const areaLoss = d3.area<ProbPoint>()
      .x(d => xScale(d.xOver))
      .y0(d => yScale(1 - d.pWin - d.pDraw))
      .y1(0)
      .curve(d3.curveMonotoneX);

    // Clear previous render
    this.svg.selectAll('*').remove();

    // Render areas
    this.svg.append('path')
      .datum(data)
      .attr('class', 'area-win')
      .attr('d', areaWin)
      .attr('fill', '#22c55e')
      .attr('opacity', 0.8);

    this.svg.append('path')
      .datum(data)
      .attr('class', 'area-draw')
      .attr('d', areaDraw)
      .attr('fill', '#6b7280')
      .attr('opacity', 0.8);

    this.svg.append('path')
      .datum(data)
      .attr('class', 'area-loss')
      .attr('d', areaLoss)
      .attr('fill', '#ef4444')
      .attr('opacity', 0.8);

    // X-axis
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d => {
        const xOver = d.valueOf();
        const innings = Math.floor(xOver / OVER_OFFSET) + 1;
        const over = xOver % OVER_OFFSET;
        return innings > 1 && over === 0 ? `Inn ${innings}` : over.toString();
      });

    this.svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${this.chartHeight})`)
      .call(xAxis)
      .selectAll('text')
      .style('fill', '#9ca3af')
      .style('font-size', '12px');

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
        .attr('fill', item.color)
        .attr('opacity', 0.8);

      g.append('text')
        .attr('x', 24)
        .attr('y', 13)
        .style('fill', '#9ca3af')
        .style('font-size', '14px')
        .text(item.label);
    });

    // Innings boundaries
    for (let i = 1; i < 4; i++) {
      const xPos = xScale(i * OVER_OFFSET);
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
}
