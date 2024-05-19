import { z } from 'zod';
import { eq } from 'drizzle-orm';
import { ilike } from "drizzle-orm";

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from '~/server/api/trpc';
import { movieProductions, movies } from '~/server/db/schema';

//** movie route */
export const movieRouter = createTRPCRouter({
  // defines purpose, protected = requires login
  byId: publicProcedure
    // validates input
    .input(z.number())
    // sets up query with context and input
    .query(({ ctx, input }) => {
      // drizzle orm select
      // https://orm.drizzle.team/docs/rqb
      return ctx.db.query.movies.findFirst({
        where: eq(movies.id, input),
        with:{
          movieProductions: {
            with:{
              production: true,
            },
          },
          reviews: true,
        }
      })
    }),

  getAll: publicProcedure.query(({ ctx }) => {
    return ctx.db.query.movies.findMany();
  }),

  searchFor: publicProcedure
  .input(z.string())
  .query(({ctx, input}) =>{
    return ctx.db.select().from(movies).where(ilike(movies.title, "%" + input + "%")).limit(10);
  })
});
