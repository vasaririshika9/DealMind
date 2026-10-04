/**
 * DealSight AI - Centralized API Configuration
 *
 * In local development, loads from frontend/.env (http://localhost:8001)
 * In production builds on Vercel, loads from frontend/.env.production (https://dealmind-cln8.onrender.com)
 */

export const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '');

if (!API_URL && typeof window !== 'undefined') {
  console.warn('[DealSight AI] VITE_API_URL is not set. Requests will use relative origin.');
}
