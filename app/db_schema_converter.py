"""
Database Schema Converter for Financial Data

Converts array-based JSON extraction to database-ready key-value format
compatible with Redis, PostgreSQL, and MongoDB.

TARGET SCHEMA (per data point):
{
    "company": "CEAT LTD",
    "document_type": "profit_loss",
    "period": "Mar-24",
    "period_type": "annual",
    "metric": "sales",
    "value": 11943.48,
    "unit": "crores",
    "source_pdf": "CEAT_Peer 3 (1).pdf"
}

This flat structure allows:
- SQL: SELECT value FROM metrics WHERE company='CEAT' AND metric='sales' AND period='Mar-24'
- MongoDB: db.metrics.find({company: "CEAT", metric: "sales"})
- Redis: GET "CEAT:profit_loss:Mar-24:sales"
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def normalize_period(period: str) -> Optional[str]:
    """Normalize period format to YYYY-MM for sorting."""
    if not period:
        return None
    
    # Handle Mar-24 format
    match = re.match(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{2})', period)
    if match:
        month_map = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
            'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
        }
        month = month_map[match.group(1)]
        year = match.group(2)
        # Assume 20XX for years < 50, else 19XX
        full_year = f"20{year}" if int(year) < 50 else f"19{year}"
        return f"{full_year}-{month}"
    
    return period


def classify_period_type(period: str) -> str:
    """Classify period as annual, quarterly, or unknown."""
    if not period:
        return "unknown"
    
    # Annual periods typically end in Mar (fiscal year end in India)
    if period.startswith("Mar-"):
        return "annual"
    
    # Quarterly periods
    if any(period.startswith(m) for m in ["Jun-", "Sep-", "Dec-"]):
        return "quarterly"
    
    return "unknown"


def extract_company_name(data: Dict) -> str:
    """Extract company name from metadata or document data."""
    # Check metadata documents
    if "documents" in data:
        for doc in data.get("documents", []):
            if doc.get("document_type") == "metadata":
                extracted = doc.get("extracted_data", {})
                if "company_name" in extracted:
                    return extracted["company_name"]
    
    # Try to extract from source_pdf
    source = data.get("source_pdf", "")
    if source:
        name = Path(source).stem
        # Clean up common suffixes
        for suffix in ["_Peer", "_Target", "_pages"]:
            name = name.split(suffix)[0]
        return name.replace("_", " ").strip()
    
    return "Unknown Company"


def convert_to_flat_records(
    extraction_result: Dict,
    company_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convert extraction result to flat database records.
    
    Each record represents a single metric for a single period.
    """
    records = []
    
    source_pdf = extraction_result.get("source_pdf", "")
    if not company_name:
        company_name = extract_company_name(extraction_result)
    
    # Process each document
    for doc in extraction_result.get("documents", []):
        doc_type = doc.get("document_type", "unknown")
        period_type_from_doc = doc.get("period_type", "unknown")
        periods = doc.get("periods", [])
        data = doc.get("data", {})
        
        # Get periods from data if not in doc
        if not periods and "periods" in data:
            periods = data.get("periods", [])
        
        # Process each metric
        for metric, values in data.items():
            if metric == "periods":
                continue
            
            # Handle array values (one per period)
            if isinstance(values, list):
                for i, value in enumerate(values):
                    if i < len(periods):
                        period = periods[i]
                        period_type = period_type_from_doc
                        
                        # Auto-classify if unknown
                        if period_type == "unknown":
                            period_type = classify_period_type(period)
                        
                        # Skip if period doesn't match expected type
                        # (helps filter out mixed data)
                        actual_type = classify_period_type(period)
                        if period_type != "unknown" and actual_type != "unknown":
                            if period_type == "annual" and actual_type != "annual":
                                continue
                            if period_type == "quarterly" and actual_type != "quarterly":
                                continue
                        
                        record = {
                            "company": company_name,
                            "document_type": doc_type,
                            "period": period,
                            "period_normalized": normalize_period(period),
                            "period_type": period_type,
                            "metric": metric,
                            "value": value,
                            "unit": "crores",  # Default for Indian financials
                            "source_pdf": Path(source_pdf).name if source_pdf else "",
                            "timestamp": datetime.now().isoformat()
                        }
                        records.append(record)
            
            # Handle scalar values (metadata)
            else:
                record = {
                    "company": company_name,
                    "document_type": doc_type,
                    "period": None,
                    "period_normalized": None,
                    "period_type": "none",
                    "metric": metric,
                    "value": values,
                    "unit": infer_unit(metric, values),
                    "source_pdf": Path(source_pdf).name if source_pdf else "",
                    "timestamp": datetime.now().isoformat()
                }
                records.append(record)
    
    return records


def infer_unit(metric: str, value: Any) -> str:
    """Infer unit based on metric name and value."""
    metric_lower = metric.lower()
    
    if "percent" in metric_lower or "ratio" in metric_lower or "opm" in metric_lower:
        return "percent"
    if "days" in metric_lower:
        return "days"
    if "shares" in metric_lower or "num_" in metric_lower:
        return "units"
    if "price" in metric_lower or "eps" in metric_lower:
        return "rupees"
    if "face_value" in metric_lower:
        return "rupees"
    if "market_cap" in metric_lower:
        return "crores"
    
    return "crores"


def generate_redis_commands(records: List[Dict]) -> List[str]:
    """Generate Redis SET commands for the records."""
    commands = []
    
    for r in records:
        company = r["company"].replace(" ", "_")
        doc_type = r["document_type"]
        period = r["period"] or "current"
        metric = r["metric"]
        value = r["value"]
        
        # Key format: COMPANY:DOC_TYPE:PERIOD:METRIC
        key = f"{company}:{doc_type}:{period}:{metric}"
        
        if value is not None:
            commands.append(f'SET "{key}" "{value}"')
    
    return commands


def generate_postgres_schema() -> str:
    """Generate PostgreSQL CREATE TABLE statement."""
    return """
-- Financial Metrics Table
CREATE TABLE IF NOT EXISTS financial_metrics (
    id SERIAL PRIMARY KEY,
    company VARCHAR(255) NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    period VARCHAR(20),
    period_normalized DATE,
    period_type VARCHAR(20),
    metric VARCHAR(100) NOT NULL,
    value DECIMAL(20, 4),
    unit VARCHAR(20),
    source_pdf VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for common queries
    CONSTRAINT unique_metric UNIQUE (company, document_type, period, metric)
);

-- Indexes for fast queries
CREATE INDEX idx_company ON financial_metrics(company);
CREATE INDEX idx_metric ON financial_metrics(metric);
CREATE INDEX idx_period ON financial_metrics(period);
CREATE INDEX idx_period_normalized ON financial_metrics(period_normalized);
CREATE INDEX idx_doc_type ON financial_metrics(document_type);

-- Sample queries:
-- SELECT value FROM financial_metrics WHERE company='CEAT LTD' AND metric='sales' AND period='Mar-24';
-- SELECT * FROM financial_metrics WHERE company='CEAT LTD' AND document_type='profit_loss' ORDER BY period_normalized;
"""


def generate_postgres_inserts(records: List[Dict]) -> List[str]:
    """Generate PostgreSQL INSERT statements."""
    inserts = []
    
    for r in records:
        value = r["value"]
        if value is None:
            value_str = "NULL"
        elif isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)
        
        period = f"'{r['period']}'" if r['period'] else "NULL"
        period_norm = f"'{r['period_normalized']}'" if r['period_normalized'] else "NULL"
        
        insert = f"""INSERT INTO financial_metrics 
            (company, document_type, period, period_normalized, period_type, metric, value, unit, source_pdf)
            VALUES ('{r['company']}', '{r['document_type']}', {period}, {period_norm}, 
                    '{r['period_type']}', '{r['metric']}', {value_str}, '{r['unit']}', '{r['source_pdf']}')
            ON CONFLICT (company, document_type, period, metric) DO UPDATE SET value = EXCLUDED.value;"""
        inserts.append(insert)
    
    return inserts


def generate_mongodb_documents(records: List[Dict]) -> List[Dict]:
    """Generate MongoDB documents."""
    # Group by company for better document structure
    companies = {}
    
    for r in records:
        company = r["company"]
        if company not in companies:
            companies[company] = {
                "company": company,
                "source_pdfs": set(),
                "last_updated": r["timestamp"],
                "metrics": {}
            }
        
        companies[company]["source_pdfs"].add(r["source_pdf"])
        
        # Create nested structure: metrics.doc_type.period.metric = value
        doc_type = r["document_type"]
        period = r["period"] or "current"
        metric = r["metric"]
        
        if doc_type not in companies[company]["metrics"]:
            companies[company]["metrics"][doc_type] = {}
        if period not in companies[company]["metrics"][doc_type]:
            companies[company]["metrics"][doc_type][period] = {}
        
        companies[company]["metrics"][doc_type][period][metric] = {
            "value": r["value"],
            "unit": r["unit"]
        }
    
    # Convert sets to lists for JSON serialization
    docs = []
    for company, data in companies.items():
        data["source_pdfs"] = list(data["source_pdfs"])
        docs.append(data)
    
    return docs


def convert_extraction_folder(folder_path: str, output_path: Optional[str] = None) -> Dict:
    """
    Convert all extraction results in a folder to database-ready format.
    """
    folder = Path(folder_path)
    
    # Look for extraction_result.json
    result_file = folder / "extraction_result.json"
    if not result_file.exists():
        raise FileNotFoundError(f"No extraction_result.json found in {folder}")
    
    with open(result_file) as f:
        extraction = json.load(f)
    
    # Convert to flat records
    records = convert_to_flat_records(extraction)
    
    # Generate outputs
    output = {
        "source_folder": str(folder),
        "record_count": len(records),
        "records": records,
        "redis_commands": generate_redis_commands(records),
        "postgres_schema": generate_postgres_schema(),
        "postgres_inserts": generate_postgres_inserts(records)[:10],  # Sample
        "mongodb_documents": generate_mongodb_documents(records)
    }
    
    # Save if output path provided
    if output_path:
        out_file = Path(output_path)
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"[SAVED] Database-ready output: {out_file}")
    
    return output


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else None
        result = convert_extraction_folder(folder, output)
        print(f"Converted {result['record_count']} records")
        print(f"\nSample Redis commands:")
        for cmd in result['redis_commands'][:5]:
            print(f"  {cmd}")
        print(f"\nPostgreSQL Schema:\n{result['postgres_schema'][:500]}...")
    else:
        print("Usage: python db_schema_converter.py <extraction_folder> [output.json]")
