/**
 * DealSight AI - Centralized API Service Configuration
 *
 * In local development, reads from frontend/.env (e.g. http://localhost:8001)
 * In production, reads from frontend/.env.production (https://dealmind-cln8.onrender.com)
 * or Vercel environment settings.
 */

export const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '');

export default API_URL;
