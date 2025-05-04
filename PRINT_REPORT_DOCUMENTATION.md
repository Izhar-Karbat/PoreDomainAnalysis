# Print-Friendly Report Functionality Documentation

This document provides information about the print-friendly report generation functionality in the Pore Domain Analysis suite. It includes code snippets and architectural details to help recreate this functionality after a git reset.

## Overview

The print-friendly report functionality converts the HTML interactive reports into print-optimized PDF reports. It handles:

1. Font embedding for consistent rendering
2. Proper page breaks and continuous flow
3. Image sizing and orientation
4. Table formatting for print media
5. Navigation and cross-referencing

## Key Files

The main files involved in the print report functionality are:

1. `pore_analysis/print_report.py` - Main script for generating print reports
2. `pore_analysis/templates/print_report_template.html` - Template for print-friendly output
3. `pore_analysis/templates/font_embedding.js` - JavaScript for embedding fonts

## Implementation Details

### Main Script (`print_report.py`)

The main script handles:
- Reading the analysis database
- Fetching all relevant plots and metrics
- Assembling print-friendly HTML
- Optional conversion to PDF

Here's a code snippet showing the general structure:

```python
def generate_print_report(run_dir, output_file=None, convert_to_pdf=False):
    """
    Generate a print-friendly report for the given run directory.
    
    Args:
        run_dir (str): Path to the run directory containing the analysis_registry.db
        output_file (str, optional): Path to save the output file. If not provided,
                                    defaults to <run_dir>/print_report.html
        convert_to_pdf (bool): Whether to convert the HTML report to PDF
    
    Returns:
        str: Path to the generated report file
    """
    # Connect to database
    db_path = os.path.join(run_dir, 'analysis_registry.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(run_dir, 'print_report.html')
    
    # Get simulation metadata
    metadata = get_simulation_metadata(conn)
    
    # Get metrics from database
    metrics = get_all_metrics(conn)
    
    # Get list of all plots
    plots = get_all_plots(conn)
    
    # Load the print report template
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
    template = env.get_template('print_report_template.html')
    
    # Render the template with data
    html_content = template.render(
        metadata=metadata,
        metrics=metrics,
        plots=plots,
        run_dir=run_dir,
        generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Write the HTML report
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    # Convert to PDF if requested
    if convert_to_pdf:
        pdf_path = output_file.replace('.html', '.pdf')
        convert_html_to_pdf(output_file, pdf_path)
        return pdf_path
    
    return output_file
```

### Print Report Template

The template is structured to optimize for printing, with CSS media queries for print:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Pore Analysis Report</title>
    <style>
        /* Common styles */
        body {
            font-family: 'Roboto', 'Arial', sans-serif;
            line-height: 1.5;
            color: #333;
        }
        
        /* Print-specific styles */
        @media print {
            @page {
                size: letter;
                margin: 0.5in;
            }
            
            /* Force page breaks */
            .page-break {
                page-break-after: always;
            }
            
            /* Avoid page breaks inside elements */
            table, figure {
                page-break-inside: avoid;
            }
            
            /* Hide navigation elements */
            nav, .no-print {
                display: none !important;
            }
            
            /* Adjust image sizes */
            img {
                max-width: 100% !important;
                max-height: 8in !important;
            }
        }
        
        /* Rest of the styles... */
    </style>
    <script>
        {% include 'font_embedding.js' %}
    </script>
</head>
<body>
    <header>
        <h1>Ion Channel Pore Analysis Report</h1>
        <div class="metadata">
            <p><strong>Simulation:</strong> {{ metadata.run_name }}</p>
            <p><strong>Generated:</strong> {{ generation_date }}</p>
        </div>
    </header>
    
    <div class="table-of-contents">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#overview">Overview</a></li>
            <li><a href="#core-analysis">Core Analysis</a></li>
            <!-- Other sections -->
        </ul>
    </div>
    
    <section id="overview" class="page-break">
        <h2>Overview</h2>
        <!-- Overview content -->
    </section>
    
    <!-- Other sections -->
    
    <!-- For each plot type, include with proper formatting -->
    {% for plot in plots %}
        {% if plot.category == 'visualization' %}
            <figure>
                <img src="{{ plot.path }}" alt="{{ plot.description }}">
                <figcaption>{{ plot.description }}</figcaption>
            </figure>
        {% endif %}
    {% endfor %}
    
    <!-- For each metric, include in appropriate tables -->
    {% for metric_group in metrics|groupby('module_name') %}
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Units</th>
                </tr>
            </thead>
            <tbody>
                {% for metric in metric_group.items %}
                    <tr>
                        <td>{{ metric.name }}</td>
                        <td>{{ metric.value|float|round(3) }}</td>
                        <td>{{ metric.units }}</td>
                    </tr>
                {% endfor %}
            </tbody>
        </table>
    {% endfor %}
</body>
</html>
```

### Font Embedding

A critical part of the print report is embedding fonts to ensure consistent rendering:

```javascript
// Font embedding script
(function() {
    // List of fonts to embed
    const fonts = [
        {
            family: 'Roboto',
            weights: [400, 700],
            styles: ['normal', 'italic']
        },
        {
            family: 'Roboto Mono',
            weights: [400],
            styles: ['normal']
        }
    ];
    
    // Load fonts using FontFace API
    fonts.forEach(font => {
        font.weights.forEach(weight => {
            font.styles.forEach(style => {
                const fontFace = new FontFace(
                    font.family,
                    `url(https://fonts.gstatic.com/s/${font.family.toLowerCase().replace(' ', '')}/v10/${font.family.toLowerCase().replace(' ', '')}-${weight}${style === 'italic' ? 'italic' : ''}.woff2) format('woff2')`,
                    { weight, style }
                );
                
                fontFace.load().then(loadedFace => {
                    document.fonts.add(loadedFace);
                }).catch(err => {
                    console.error(`Failed to load font: ${font.family} ${weight} ${style}`, err);
                });
            });
        });
    });
})();
```

## Command-Line Interface

The functionality is typically invoked via the command line:

```python
def main():
    parser = argparse.ArgumentParser(description='Generate a print-friendly report from analysis results')
    parser.add_argument('run_dir', help='Path to the run directory')
    parser.add_argument('--output', '-o', help='Output file path (default: <run_dir>/print_report.html)')
    parser.add_argument('--pdf', action='store_true', help='Convert the report to PDF')
    args = parser.parse_args()
    
    output_file = generate_print_report(args.run_dir, args.output, args.pdf)
    print(f"Report generated: {output_file}")

if __name__ == '__main__':
    main()
```

## PDF Conversion

For PDF conversion, the code typically uses either WeasyPrint or a headless browser:

```python
def convert_html_to_pdf(html_path, pdf_path):
    """
    Convert HTML to PDF using WeasyPrint.
    
    Args:
        html_path (str): Path to the HTML file
        pdf_path (str): Path to save the PDF file
    """
    try:
        from weasyprint import HTML
        HTML(html_path).write_pdf(pdf_path)
    except ImportError:
        # Fallback to using a headless browser
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            
            driver = webdriver.Chrome(options=options)
            driver.get(f'file://{os.path.abspath(html_path)}')
            
            # Wait for any scripts to load
            time.sleep(2)
            
            # Set PDF settings
            pdf_options = {
                'landscape': False,
                'paperWidth': 8.5,
                'paperHeight': 11,
                'printBackground': True,
                'marginTop': 0.5,
                'marginBottom': 0.5,
                'marginLeft': 0.5,
                'marginRight': 0.5
            }
            
            # Print to PDF
            result = driver.execute_cdp_cmd('Page.printToPDF', pdf_options)
            with open(pdf_path, 'wb') as f:
                f.write(base64.b64decode(result['data']))
                
            driver.quit()
        except Exception as e:
            print(f"Failed to convert HTML to PDF: {e}")
            print("Please manually print the HTML file to PDF.")
```

## Integration with HTML Report

The print report functionality is integrated with the existing HTML report generation:

```python
def generate_reports(run_dir, generate_html=True, generate_print=False, convert_to_pdf=False):
    """
    Generate reports for the given run directory.
    
    Args:
        run_dir (str): Path to the run directory
        generate_html (bool): Whether to generate the interactive HTML report
        generate_print (bool): Whether to generate the print-friendly report
        convert_to_pdf (bool): Whether to convert the print report to PDF
    
    Returns:
        dict: Paths to the generated reports
    """
    reports = {}
    
    if generate_html:
        from pore_analysis.html import generate_html_report
        html_path = generate_html_report(run_dir)
        reports['html'] = html_path
    
    if generate_print:
        from pore_analysis.print_report import generate_print_report
        print_path = generate_print_report(run_dir, convert_to_pdf=convert_to_pdf)
        reports['print'] = print_path
    
    return reports
```

## Styling Considerations for Print Media

Special considerations for print-friendly styling:

1. Use physical units (inches, cm) instead of pixels
2. Ensure proper page breaks with `page-break-*` properties
3. Set appropriate page margins and sizes
4. Optimize images for print resolution
5. Use printer-friendly colors (avoid light colors that don't print well)

## Database Queries

The print report typically uses these database queries to get content:

```python
def get_all_metrics(conn):
    """Get all metrics from the database."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT module_name, metric_name, value, units
        FROM metrics
        ORDER BY module_name, metric_name
    """)
    return cursor.fetchall()

def get_all_plots(conn):
    """Get all visualization plots from the database."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.module_name, p.product_type, p.category, p.subcategory, 
               p.description, p.relative_path
        FROM analysis_products p
        WHERE p.category = 'visualization'
        ORDER BY p.module_name, p.subcategory
    """)
    
    plots = []
    for row in cursor.fetchall():
        # Convert relative path to absolute
        abs_path = os.path.join(run_dir, row['relative_path'])
        plots.append({
            'module_name': row['module_name'],
            'type': row['product_type'],
            'category': row['category'],
            'subcategory': row['subcategory'],
            'description': row['description'],
            'path': abs_path
        })
    
    return plots
```

## Additional Resources

For rebuilding this functionality, consider:

1. The W3C CSS Paged Media spec
2. WeasyPrint documentation
3. Jinja2 templating documentation
4. MDN Web Docs on CSS for print media

This documentation should help in recreating the print-friendly report functionality after a git reset.