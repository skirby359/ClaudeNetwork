"""Page-level logging helpers. Import and call at the top of each page."""

import logging
import traceback
import functools

import streamlit as st


def log_page_entry(page_name: str):
    """Log that a page was entered. Call at the top of each page."""
    logger = logging.getLogger(f"page.{page_name}")
    logger.info(f"Page entered: {page_name}")
    return logger


def log_page_error(page_name: str, error: Exception):
    """Log a page-level error with full traceback."""
    logger = logging.getLogger(f"page.{page_name}")
    logger.error(f"Page {page_name} error: {error}")
    logger.debug(traceback.format_exc())


def safe_cached(page_name: str):
    """Decorator for page-level cached functions that logs errors.

    Usage:
        @safe_cached("page_25")
        @st.cache_data(ttl=3600)
        def _cached_analysis(start_date, end_date):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(f"page.{page_name}")
            fname = func.__name__
            try:
                logger.info(f"  {fname}() called with args={[str(a)[:50] for a in args]}")
                result = func(*args, **kwargs)
                # Log result shape
                if hasattr(result, '__len__'):
                    logger.info(f"  {fname}() returned {type(result).__name__} len={len(result)}")
                elif isinstance(result, tuple):
                    shapes = []
                    for r in result:
                        if hasattr(r, '__len__'):
                            shapes.append(f"{type(r).__name__}({len(r)})")
                        else:
                            shapes.append(type(r).__name__)
                    logger.info(f"  {fname}() returned tuple: {', '.join(shapes)}")
                elif isinstance(result, dict):
                    logger.info(f"  {fname}() returned dict keys={list(result.keys())[:10]}")
                else:
                    logger.info(f"  {fname}() returned {type(result).__name__}")
                return result
            except Exception as e:
                logger.error(f"  {fname}() FAILED: {e}")
                logger.error(traceback.format_exc())
                raise
        return wrapper
    return decorator
